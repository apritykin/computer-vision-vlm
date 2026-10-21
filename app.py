"""Object Detection vs. Vision-Language Model: a local webcam demo.

Two models look at the same webcam image and answer two different questions:

    Object detection (YOLO)  ->  "What is there, and where is it?"
                                  drawn as bounding boxes on the video
    Vision-language (Qwen-VL) ->  "What is happening in the scene?"
                                  written as one sentence in the black footer

The detector runs on every frame. Qwen-VL only runs when you press SPACE, in a
background thread, so the video never freezes while it thinks.

Keys:
    SPACE  capture the current frame and ask Qwen-VL to describe it
    d      toggle the object-detection overlay
    v      toggle Qwen-VL on/off
    q      quit (Esc or closing the window also works)

Everything runs locally. No cloud APIs.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# CONFIGURATION
# Each value can be overridden with an environment variable or a command-line
# flag (run `python app.py --help`).
# =============================================================================

# Qwen vision-language model (any Hugging Face Qwen-VL "Instruct" checkpoint).
#   Qwen/Qwen3-VL-2B-Instruct   ~4 GB   fast, the default
#   Qwen/Qwen3-VL-4B-Instruct   ~8 GB   more detailed, slower
#   Qwen/Qwen2.5-VL-3B-Instruct ~7 GB   older generation
VLM_MODEL = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen3-VL-2B-Instruct")

# Object-detection weights. Ultralytics downloads them on first run.
#   yolo11n.pt (nano, fastest) / yolo11s.pt (small) / yolo11m.pt (medium)
DETECTOR_MODEL = os.environ.get("DETECTOR_MODEL", "yolo11n.pt")

# Hide detections the detector is less sure about than this (0.0 - 1.0).
DETECTION_CONFIDENCE = float(os.environ.get("DETECTION_CONFIDENCE", "0.4"))

VLM_PROMPT = (
    "Describe what is happening in this image in exactly one concise sentence "
    "of no more than 20 words. "
    "Focus on the people, important objects, actions, and spatial relationships "
    "that are visually apparent. Do not speculate about anything that cannot be seen."
)

# Longest side (in pixels) of the image handed to Qwen-VL. Smaller = faster.
VLM_IMAGE_SIZE = 768

WINDOW_NAME = "Object Detection vs. Vision-Language Model"

# =============================================================================


def pick_torch_device(requested: str) -> str:
    """Resolve "auto" to the best local accelerator: CUDA, Apple MPS, or CPU."""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def short_error(error: BaseException) -> str:
    """First line of an exception, short enough for the footer."""
    first_line = (str(error).strip().splitlines() or [""])[0]
    return f"{type(error).__name__}: {first_line}"[:120]


def is_out_of_memory(error: BaseException) -> bool:
    return isinstance(error, MemoryError) or "out of memory" in str(error).lower()


# -----------------------------------------------------------------------------
# Object detection
# -----------------------------------------------------------------------------


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels


class YoloDetector:
    """Ultralytics YOLO wrapper.

    To swap in a different detector, write a class with the same `name`
    attribute and `detect(frame) -> list[Detection]` method.
    """

    def __init__(self, weights: str, confidence: float, device: str):
        from ultralytics import YOLO

        self.model = YOLO(weights)  # downloads the weights if they are missing
        self.confidence = confidence
        self.device = device
        self.name = os.path.splitext(os.path.basename(weights))[0].upper()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            frame, conf=self.confidence, device=self.device, verbose=False
        )[0]
        detections = []
        for box, conf, cls in zip(
            result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
        ):
            x1, y1, x2, y2 = (int(v) for v in box)
            detections.append(Detection(result.names[int(cls)], conf, (x1, y1, x2, y2)))
        return detections


# -----------------------------------------------------------------------------
# Vision-language model
# -----------------------------------------------------------------------------


def to_one_sentence(text: str) -> str:
    """Collapse whatever the model returned into exactly one clean sentence."""
    text = " ".join(text.split()).strip().strip("\"'“”")
    match = re.match(r"(.+?[.!?])(?=\s|$)", text)
    sentence = (match.group(1) if match else text).strip()
    if not sentence:
        return ""
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence[0].upper() + sentence[1:]


class QwenVL:
    """Loads a Qwen-VL checkpoint with Hugging Face transformers and describes images."""

    def __init__(self, model_id: str, device: str, max_new_tokens: int):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = {"cuda": torch.bfloat16, "mps": torch.bfloat16}.get(self.device, torch.float32)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, dtype=dtype)
        self.model.to(self.device).eval()

    def describe(self, frame_bgr: np.ndarray) -> str:
        import torch

        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image.thumbnail((VLM_IMAGE_SIZE, VLM_IMAGE_SIZE))

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": VLM_PROMPT}]}
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return to_one_sentence(text)

    def free_memory(self) -> None:
        import torch

        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()


class VlmWorker:
    """Runs Qwen-VL in a background thread, one request at a time.

    The main loop only ever calls `request()` and `snapshot()`, both of which
    return immediately, so the webcam keeps updating while the model works.
    """

    def __init__(self, vlm: QwenVL):
        self.vlm = vlm
        self.lock = threading.Lock()
        self.busy = False
        self.loaded = False
        self.status = "loading"  # loading | ready | analyzing | error
        self.summary = "Press SPACE to describe the scene."
        self.error = ""
        self.started_at = 0.0
        self.last_seconds: float | None = None
        self.pending: np.ndarray | None = None  # frame currently being analyzed
        self.described: np.ndarray | None = None  # frame the current summary describes
        self.described_id = 0  # bumps every time `described` changes

    def start_loading(self) -> None:
        self._start(None)

    def request(self, frame: np.ndarray) -> bool:
        """Queue `frame` for description. Returns False if a request is already running."""
        return self._start(frame.copy())

    def _start(self, frame: np.ndarray | None) -> bool:
        with self.lock:
            if self.busy:
                return False
            self.busy = True
            self.error = ""
            if frame is not None:
                self.pending = frame
                self.started_at = time.perf_counter()
            self.status = "analyzing" if (frame is not None and self.loaded) else "loading"
        threading.Thread(target=self._run, args=(frame,), daemon=True).start()
        return True

    def _run(self, frame: np.ndarray | None) -> None:
        try:
            if not self.loaded:
                print(f"Loading {self.vlm.model_id} on {self.vlm.device} ...")
                self.vlm.load()
                self.loaded = True
                print("Qwen-VL ready.")
                with self.lock:
                    self.status = "analyzing" if frame is not None else "ready"
                    if frame is not None:
                        self.started_at = time.perf_counter()  # don't count load time
            if frame is not None:
                sentence = self.vlm.describe(frame)
                with self.lock:
                    self.last_seconds = time.perf_counter() - self.started_at
                    self.summary = sentence or "(Qwen-VL returned an empty description.)"
                    self.described = frame
                    self.described_id += 1
                print(f"[Qwen-VL {self.last_seconds:.1f}s] {self.summary}")
            with self.lock:
                self.status = "ready"
        except Exception as error:  # never let a model problem take down the app
            if is_out_of_memory(error):
                message = "Not enough memory for this model. Try a smaller QWEN_VL_MODEL."
                self.vlm.free_memory()
            elif not self.loaded:
                message = f"Could not load {self.vlm.model_id}: {short_error(error)}"
            else:
                message = f"Inference failed ({short_error(error)}). Press SPACE to try again."
            print(f"Qwen-VL error: {type(error).__name__}: {error}", file=sys.stderr)
            with self.lock:
                self.status = "error"
                self.error = message
        finally:
            with self.lock:
                self.busy = False

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "summary": self.summary,
                "error": self.error,
                "elapsed": time.perf_counter() - self.started_at,
                "last_seconds": self.last_seconds,
                "described": self.described,
                "described_id": self.described_id,
            }


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
AMBER = (0, 190, 255)  # OpenCV colors are (blue, green, red)
RED = (80, 80, 255)

# Bright, distinct colors so neighbouring boxes are easy to tell apart.
BOX_COLORS = [
    (255, 178, 29), (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0), (147, 69, 52),
    (255, 115, 100), (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203),
]

CV_FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(label: str) -> tuple[int, int, int]:
    return BOX_COLORS[sum(map(ord, label)) % len(BOX_COLORS)]


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    """Draw one box and a "LABEL 92%" tag per detection."""
    width = frame.shape[1]
    thickness = max(2, width // 640)
    scale = max(0.45, width / 1280 * 0.6)
    for det in detections:
        x1, y1, x2, y2 = det.box
        color = color_for(det.label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        tag = f"{det.label.upper()} {det.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(tag, CV_FONT, scale, 1)
        pad = 5
        top = y1 - th - baseline - pad * 2
        if top < 0:  # no room above the box: put the tag just inside it
            top = y1
        cv2.rectangle(frame, (x1, top), (x1 + tw + pad * 2, top + th + baseline + pad * 2), color, cv2.FILLED)
        brightness = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        text_color = BLACK if brightness > 140 else WHITE
        cv2.putText(frame, tag, (x1 + pad, top + pad + th), CV_FONT, scale, text_color, 1, cv2.LINE_AA)


def draw_pill(frame: np.ndarray, text: str, x: int, y: int, align_right: bool = False) -> None:
    """Small translucent status label, e.g. "OBJECT DETECTION  YOLO11N  30 FPS"."""
    scale = max(0.42, frame.shape[1] / 1280 * 0.5)
    (tw, th), baseline = cv2.getTextSize(text, CV_FONT, scale, 1)
    pad = 8
    if align_right:
        x -= tw + pad * 2
    x1, y1, x2, y2 = x, y, x + tw + pad * 2, y + th + baseline + pad * 2
    roi = frame[y1:y2, x1:x2]
    roi[:] = (roi * 0.45).astype(np.uint8)
    cv2.putText(frame, text, (x + pad, y + pad + th), CV_FONT, scale, WHITE, 1, cv2.LINE_AA)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """A clean TrueType font if one is installed, otherwise Pillow's built-in one."""
    candidates = [
        ("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 0),  # macOS
        ("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
         else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
        ("Arial Bold.ttf" if bold else "Arial.ttf", 0),
    ]
    for path, index in candidates:
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default(size)


def wrap_text(text: str, font, max_width: int, max_lines: int = 2) -> list[str]:
    """Greedy word wrap; if it still doesn't fit, end the last line with an ellipsis."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if font.getlength(trial) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        last = " ".join(lines[max_lines - 1 :])
        while last and font.getlength(last + "…") > max_width:
            last = last.rsplit(" ", 1)[0] if " " in last else last[:-1]
        lines = lines[: max_lines - 1] + [last.rstrip(" ,;") + "…"]
    return lines


class Footer:
    """The solid black "VISION SUMMARY" bar under the video.

    Rendering text with Pillow looks much nicer than OpenCV's built-in fonts.
    The bar is cached and only redrawn when its content changes.
    """

    def __init__(self, width: int):
        self.width = width
        unit = width / 1280
        self.height = int(150 * unit)
        self.pad = int(26 * unit)
        self.label_font = load_font(int(17 * unit), bold=True)
        self.text_font = load_font(int(30 * unit))
        self._key = None
        self._image: np.ndarray | None = None

    def render(self, summary: str, status: str, status_color, thumb: np.ndarray | None, thumb_id: int) -> np.ndarray:
        key = (summary, status, status_color, thumb_id)
        if key == self._key and self._image is not None:
            return self._image

        bar = np.zeros((self.height, self.width, 3), np.uint8)
        text_left = self.pad

        # Thumbnail of the exact frame the sentence describes, so viewers can
        # see Qwen-VL describes one captured moment, not the live video.
        if thumb is not None:
            th = self.height - self.pad * 2
            tw = int(th * thumb.shape[1] / thumb.shape[0])
            bar[self.pad : self.pad + th, self.pad : self.pad + tw] = cv2.resize(thumb, (tw, th))
            cv2.rectangle(bar, (self.pad, self.pad), (self.pad + tw - 1, self.pad + th - 1), (90, 90, 90), 1)
            text_left = self.pad * 2 + tw

        # Pillow treats the channels as RGB; we keep them BGR on both sides,
        # so colors stay in OpenCV's (blue, green, red) order throughout.
        image = Image.fromarray(bar)
        draw = ImageDraw.Draw(image)
        label_y = self.pad - 2
        draw.text((text_left, label_y), "VISION SUMMARY", font=self.label_font, fill=(255, 255, 255))

        if status:
            w = self.label_font.getlength(status)
            draw.text((self.width - self.pad - w, label_y), status, font=self.label_font, fill=status_color)

        max_width = self.width - text_left - self.pad
        line_height = int(self.text_font.size * 1.25)
        y = label_y + int(self.label_font.size * 1.9)
        for line in wrap_text(summary, self.text_font, max_width):
            draw.text((text_left, y), line, font=self.text_font, fill=(255, 255, 255))
            y += line_height

        self._image = np.asarray(image)
        self._key = key
        return self._image


def footer_status(vlm_state: dict | None, vlm_enabled: bool) -> tuple[str, tuple[int, int, int]]:
    if vlm_state is None or not vlm_enabled:
        return "QWEN-VL OFF  ·  PRESS V", GRAY
    status = vlm_state["status"]
    if status == "loading":
        return "LOADING QWEN-VL…", AMBER
    if status == "analyzing":
        dots = "." * (1 + int(vlm_state["elapsed"] * 2.5) % 3)
        return f"ANALYZING{dots:<3}  {vlm_state['elapsed']:.1f}s", AMBER
    if status == "error":
        return "QWEN-VL ERROR", RED
    if vlm_state["last_seconds"] is not None:
        return f"QWEN-VL  ·  {vlm_state['last_seconds']:.1f}s", GRAY
    return "QWEN-VL READY", GRAY


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vlm-model", default=VLM_MODEL, help=f"Qwen-VL checkpoint (env QWEN_VL_MODEL, default {VLM_MODEL})")
    parser.add_argument("--vlm-device", default=os.environ.get("VLM_DEVICE", "auto"), help="auto, mps, cuda, or cpu")
    parser.add_argument("--max-new-tokens", type=int, default=60, help="Upper limit on the sentence length in tokens")
    parser.add_argument("--no-vlm", action="store_true", help="Start with Qwen-VL turned off (press V to load it)")
    parser.add_argument("--detector", default=DETECTOR_MODEL, help=f"YOLO weights (env DETECTOR_MODEL, default {DETECTOR_MODEL})")
    parser.add_argument("--detector-device", default=os.environ.get("DETECTOR_DEVICE", "cpu"), help="cpu (default), mps, or cuda")
    parser.add_argument("--conf", type=float, default=DETECTION_CONFIDENCE, help=f"Detection confidence threshold (env DETECTION_CONFIDENCE, default {DETECTION_CONFIDENCE})")
    parser.add_argument("--camera", type=int, default=int(os.environ.get("CAMERA_INDEX", "0")), help="Webcam index (default 0)")
    parser.add_argument("--width", type=int, default=1280, help="Requested webcam width")
    parser.add_argument("--height", type=int, default=720, help="Requested webcam height")
    parser.add_argument("--no-mirror", action="store_true", help="Don't mirror the webcam image")
    return parser.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture | None:
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        return None
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # The first frames from some webcams are empty while they warm up.
    for _ in range(30):
        ok, frame = camera.read()
        if ok and frame is not None and frame.size:
            return camera
        time.sleep(0.05)
    camera.release()
    return None


def main() -> int:
    args = parse_args()

    camera = open_camera(args.camera, args.width, args.height)
    if camera is None:
        print(
            f"Could not open webcam {args.camera}.\n"
            "  - Is another app (Zoom, Teams, FaceTime...) using it?\n"
            "  - On macOS, allow camera access for your terminal in\n"
            "    System Settings > Privacy & Security > Camera, then run again.\n"
            "  - Try another camera with --camera 1",
            file=sys.stderr,
        )
        return 1

    # Object detection: if it can't load, keep going with just the video + VLM.
    detector = None
    try:
        print(f"Loading object detector {args.detector} ...")
        detector = YoloDetector(args.detector, args.conf, args.detector_device)
    except Exception as error:
        print(f"Object detector unavailable: {type(error).__name__}: {error}", file=sys.stderr)

    # Qwen-VL loads in the background so the window appears right away.
    vlm_enabled = not args.no_vlm
    worker = None
    vlm_problem = ""
    try:
        worker = VlmWorker(QwenVL(args.vlm_model, pick_torch_device(args.vlm_device), args.max_new_tokens))
        if vlm_enabled:
            worker.start_loading()
    except Exception as error:  # e.g. torch not installed
        print(f"Qwen-VL unavailable: {type(error).__name__}: {error}", file=sys.stderr)
        vlm_problem = short_error(error)

    show_boxes = True
    fps = 0.0
    last_time = time.perf_counter()
    bad_frames = 0
    flash_until = 0.0
    footer: Footer | None = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    print("Running. SPACE = describe scene, D = boxes, V = Qwen-VL, Q = quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None or frame.size == 0:
                bad_frames += 1
                if bad_frames > 100:  # ~a few seconds of nothing: the camera is gone
                    print("Lost the webcam feed.", file=sys.stderr)
                    break
                time.sleep(0.02)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                continue
            bad_frames = 0
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)  # mirror, like looking into a mirror
            if footer is None or footer.width != frame.shape[1]:
                footer = Footer(frame.shape[1])

            # Keep an untouched copy: this is what Qwen-VL sees when SPACE is pressed.
            clean_frame = frame

            # --- Object detection: every frame, drawn as boxes -------------------
            detections: list[Detection] = []
            if detector is not None:
                try:
                    detections = detector.detect(frame)
                except Exception as error:
                    print(f"Detection failed: {error}", file=sys.stderr)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6)) if fps else 1.0 / max(now - last_time, 1e-6)
            last_time = now

            view = frame.copy()
            if show_boxes:
                draw_detections(view, detections)

            if detector is None:
                hud = "OBJECT DETECTION UNAVAILABLE"
            elif show_boxes:
                hud = f"OBJECT DETECTION  {detector.name}  {fps:.0f} FPS"
            else:
                hud = "OBJECT DETECTION HIDDEN  (PRESS D)"
            draw_pill(view, hud, 16, 16)

            if now < flash_until:  # brief white flash = "frame captured"
                strength = (flash_until - now) / 0.25
                cv2.addWeighted(view, 1 - 0.5 * strength, np.full_like(view, 255), 0.5 * strength, 0, dst=view)

            # --- Vision summary footer: latest Qwen-VL sentence -----------------
            state = worker.snapshot() if worker is not None else None
            status, status_color = footer_status(state, vlm_enabled)
            if state is None:
                summary = f"Qwen-VL is unavailable: {vlm_problem}"
            elif state["status"] == "error":
                summary = state["error"]
            elif not vlm_enabled:
                summary = state["summary"] if state["last_seconds"] is not None else "Qwen-VL is off. Press V to turn it on."
            else:
                summary = state["summary"]
            thumb, thumb_id = (state["described"], state["described_id"]) if state else (None, 0)
            bar = footer.render(summary, status, status_color, thumb, thumb_id)
            cv2.imshow(WINDOW_NAME, np.vstack([view, bar]))

            # --- Keys ------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" ") and worker is not None and vlm_enabled:
                if worker.request(clean_frame):
                    flash_until = time.perf_counter() + 0.25
                else:
                    print("Qwen-VL is still busy; wait for the current description.")
            elif key in (ord("d"), ord("D")):
                show_boxes = not show_boxes
            elif key in (ord("v"), ord("V")) and worker is not None:
                vlm_enabled = not vlm_enabled
                if vlm_enabled and not worker.loaded:
                    worker.start_loading()

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break  # the user closed the window
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # Exit immediately even if Qwen-VL is mid-generation in its background thread.
    os._exit(code)
