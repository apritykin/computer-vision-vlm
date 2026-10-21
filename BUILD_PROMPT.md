# Build Prompt: Object Detection vs. Vision-Language Model

> Copy everything below the line into your coding agent (Claude Code, Cursor,
> Codex, etc.) from inside an empty folder. The agent should build the whole
> project from this description.

---

## Goal

Build **Object Detection vs. Vision-Language Model**: a local Python app that
shows the webcam feed in a window and puts two kinds of computer vision side by
side on the same image:

- **Object detection (YOLO)** runs on every frame and draws labeled bounding
  boxes: *what is there, and where is it?*
- **A vision-language model (Qwen-VL)** runs only when the user presses
  `SPACE`. It describes the captured frame in one sentence, shown in a black
  footer under the video: *what is happening?*

Neither model helps the other. Each one looks at the image on its own.

```
                ┌─► YOLO (every frame) ──────────► bounding boxes drawn on the video
WEBCAM ─────────┤
                └─► SPACE ─► frame copy ─► Qwen-VL (background thread) ─► one sentence in the footer
```

Everything runs locally. No cloud APIs, accounts, or API keys. The only network
access is the first-run download of the YOLO weights (~5 MB) and the Qwen-VL
checkpoint (~4 GB, into the Hugging Face cache).

## Constraints

- **Language:** Python 3.11+
- **Dependencies:** put exactly this in `requirements.txt`, comments included:
  ```
  # Webcam capture, drawing, and the app window
  opencv-python>=4.8
  numpy>=1.24
  pillow>=10.1

  # Object detection (YOLO). Downloads its weights automatically on first run.
  ultralytics>=8.3

  # Vision-language model (Qwen-VL), run locally with Hugging Face transformers.
  # Qwen3-VL needs transformers 4.57 or newer.
  torch>=2.4
  transformers>=4.57
  ```
  Use only the standard library beyond these (`argparse`, `os`, `re`, `sys`,
  `threading`, `time`, `dataclasses`).
- **Files:** exactly these:
  ```
  computer-vision-vlm/
  ├── app.py            # the whole app, ~640 lines
  ├── requirements.txt
  ├── README.md
  └── .gitignore
  ```
  `.gitignore` contents:
  ```
  # Python
  __pycache__/
  *.py[cod]
  .venv/
  venv/
  *.egg-info/
  .pytest_cache/
  .mypy_cache/
  .ruff_cache/

  # Local environment overrides (QWEN_VL_MODEL, VLM_DEVICE, ...)
  .env

  # Downloaded model weights (re-downloaded automatically)
  *.pt
  *.onnx
  *.safetensors

  # Ultralytics output folder, if predictions are ever saved
  runs/

  # OS / editor
  .DS_Store
  Thumbs.db
  .vscode/
  .idea/
  ```
- One script. Plain functions plus four classes (`YoloDetector`, `QwenVL`,
  `VlmWorker`, `Footer`) and one dataclass (`Detection`). No frameworks or
  extra modules. Split `app.py` into sections with banner comments:
  CONFIGURATION (a `# ====` block), then `# ----` banners for Object
  detection, Vision-language model, Drawing, Main loop.
- Import `torch`, `transformers` and `ultralytics` **lazily inside the methods
  that use them**, so the window can open and the app can report a friendly
  error if one of them is missing or broken.
- Use OpenCV for capture, boxes and the window; use Pillow only to render the
  footer text (it looks much nicer than OpenCV's Hershey fonts). Colors are
  BGR everywhere, including the Pillow footer (keep channels BGR on both sides
  rather than converting).
- Use `from __future__ import annotations`.

## `app.py` specification

### Module docstring
Title: `Object Detection vs. Vision-Language Model: a local webcam demo.` Show
the two models and the question each answers (YOLO → "What is there, and where
is it?", drawn as boxes; Qwen-VL → "What is happening in the scene?", one
sentence in the black footer). Say the detector runs every frame and Qwen-VL
only on `SPACE`, in a background thread, so the video never freezes. List the
keys (`SPACE`, `d`, `v`, `q` + Esc/close window). End with "Everything runs
locally. No cloud APIs." The docstring doubles as the `--help` description.

### Configuration
Each value can be overridden by an env var or a CLI flag:

| Name | Default | Env var | Notes |
|------|---------|---------|-------|
| `VLM_MODEL` | `Qwen/Qwen3-VL-2B-Instruct` | `QWEN_VL_MODEL` | comment lists 2B (~4 GB, fast), `Qwen3-VL-4B-Instruct` (~8 GB), `Qwen2.5-VL-3B-Instruct` (~7 GB, older) |
| `DETECTOR_MODEL` | `yolo11n.pt` | `DETECTOR_MODEL` | comment lists n / s / m variants |
| `DETECTION_CONFIDENCE` | `0.4` | `DETECTION_CONFIDENCE` | hide less-confident boxes |
| `VLM_IMAGE_SIZE` | `768` | — | longest side of the image handed to Qwen-VL |
| `WINDOW_NAME` | `"Object Detection vs. Vision-Language Model"` | — | |

`VLM_PROMPT`, verbatim:
> Describe what is happening in this image in exactly one concise sentence of
> no more than 20 words. Focus on the people, important objects, actions, and
> spatial relationships that are visually apparent. Do not speculate about
> anything that cannot be seen.

### Helpers
- `pick_torch_device(requested)`: return `requested` unless it's `"auto"`;
  then `cuda` if available, else `mps` if available, else `cpu`.
- `short_error(error)`: `"<TypeName>: <first line of message>"`, cut to 120
  chars, for the footer.
- `is_out_of_memory(error)`: `MemoryError` or `"out of memory"` in the message.

### Object detection
`@dataclass Detection`: `label: str`, `confidence: float`,
`box: tuple[int, int, int, int]` (x1, y1, x2, y2 pixels).

`YoloDetector(weights, confidence, device)`: wraps `ultralytics.YOLO`
(which downloads missing weights). `name` = weights file name without the
extension, upper-cased (e.g. `YOLO11N`). `detect(frame)` calls
`model.predict(frame, conf=..., device=..., verbose=False)[0]` and zips
`boxes.xyxy`, `boxes.conf`, `boxes.cls` into `Detection`s with
`result.names[cls]` as the label. The docstring says any class with the same
`name` and `detect(frame) -> list[Detection]` can replace it.

### Vision-language model
**`to_one_sentence(text)`**: collapse whitespace, strip straight and curly
quotes, keep only the first sentence (regex `(.+?[.!?])(?=\s|$)`), add a
period if it doesn't end in `.!?`, capitalize the first letter. Return `""`
for empty input.

**`QwenVL(model_id, device, max_new_tokens)`**:
- `load()`: dtype `bfloat16` on cuda/mps, else `float32`.
  `AutoProcessor.from_pretrained(model_id)` and
  `AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)`, then
  `.to(device).eval()`.
- `describe(frame_bgr)`: BGR→RGB into a PIL image, `thumbnail` to
  `VLM_IMAGE_SIZE`. Build one user message with content
  `[{"type": "image"}, {"type": "text", "text": VLM_PROMPT}]`, run
  `processor.apply_chat_template(..., add_generation_prompt=True)`, then
  `processor(text=[prompt], images=[image], return_tensors="pt").to(device)`.
  Under `torch.inference_mode()`, `generate(..., max_new_tokens, do_sample=False)`.
  Decode **only the new tokens** (slice off the input length),
  `skip_special_tokens=True`, and return `to_one_sentence(...)`.
- `free_memory()`: `torch.mps.empty_cache()` or `torch.cuda.empty_cache()`.

**`VlmWorker(vlm)`**: runs Qwen-VL in a daemon thread, one request at a time.
The main loop only calls `request()` and `snapshot()`, which return
immediately. Everything shared is guarded by a `threading.Lock`.
- State: `busy`, `loaded`, `status` (`loading | ready | analyzing | error`),
  `summary` (initially `"Press SPACE to describe the scene."`), `error`,
  `started_at`, `last_seconds` (None until the first result), `pending`,
  `described` (the frame the current summary describes), `described_id`
  (bumped whenever `described` changes).
- `start_loading()` = `_start(None)`; `request(frame)` = `_start(frame.copy())`.
- `_start(frame)`: under the lock, return `False` if `busy`. Otherwise set
  busy, clear `error`, record `pending` and `started_at` if there's a frame,
  set status `analyzing` if loaded and there's a frame, else `loading`. Start
  the thread and return `True`.
- `_run(frame)`: load the model if needed (print `Loading <id> on <device> ...`
  and `Qwen-VL ready.`), and reset `started_at` after loading so load time
  isn't counted. If there's a frame, describe it and store `last_seconds`,
  `summary` (or `"(Qwen-VL returned an empty description.)"`), `described`,
  `described_id += 1`. Print `[Qwen-VL 1.2s] <sentence>`. Set status `ready`.
  **Catch every exception** so a model problem never takes down the app:
  out-of-memory → `"Not enough memory for this model. Try a smaller QWEN_VL_MODEL."`
  (and `free_memory()`); failed load → `"Could not load <id>: <short_error>"`;
  otherwise → `"Inference failed (<short_error>). Press SPACE to try again."`.
  Print the full error to stderr, set status `error`. Always clear `busy` in
  `finally`.
- `snapshot()`: a dict of `status`, `summary`, `error`, `elapsed` (now −
  `started_at`), `last_seconds`, `described`, `described_id`, read under the lock.

### Drawing
- Colors (BGR): `WHITE`, `BLACK`, `GRAY (150,150,150)`, `AMBER (0,190,255)`,
  `RED (80,80,255)`. `BOX_COLORS`: 15 bright, distinct colors. Font:
  `cv2.FONT_HERSHEY_SIMPLEX`.
- `color_for(label)`: stable color per class, `BOX_COLORS[sum(map(ord, label)) % len]`.
- `draw_detections(frame, detections)`: scale line thickness
  (`max(2, width // 640)`) and font (`max(0.45, width/1280*0.6)`) with frame
  width. Per detection: an anti-aliased box and a filled tag `"PERSON 92%"`
  above the box (inside it if there's no room above). Text is black or white
  depending on the tag color's brightness (`0.114B + 0.587G + 0.299R > 140`).
- `draw_pill(frame, text, x, y, align_right=False)`: a small label that darkens
  its region to 45% and writes white text on top.
- `load_font(size, bold)`: try Helvetica Neue (macOS `.ttc`, index 1 for bold),
  Segoe UI (Windows), DejaVu Sans (Linux), Arial; fall back to
  `ImageFont.load_default(size)`.
- `wrap_text(text, font, max_width, max_lines=2)`: greedy word wrap using
  `font.getlength`; if it overflows, trim the last line and end it with `…`.
- **`Footer(width)`**: the solid black "VISION SUMMARY" bar under the video.
  Everything scales with `unit = width / 1280`: height 150, padding 26, a bold
  17 px label font and a 30 px text font. `render(summary, status,
  status_color, thumb, thumb_id)` **caches** its output and only redraws when
  that tuple (with `thumb_id` standing in for the image) changes. Layout: a
  thumbnail of the exact frame the sentence describes on the left (with a thin
  gray border, so viewers see Qwen-VL describes one captured moment, not the
  live video), then `VISION SUMMARY` top-left, the status right-aligned in its
  color, and the sentence wrapped to two lines below.
- `footer_status(vlm_state, vlm_enabled) -> (text, color)`:
  - off or unavailable → `QWEN-VL OFF  ·  PRESS V`, gray
  - loading → `LOADING QWEN-VL…`, amber
  - analyzing → `ANALYZING` + 1–3 animated dots (padded to width 3) + `  1.2s` elapsed, amber
  - error → `QWEN-VL ERROR`, red
  - ready with a previous result → `QWEN-VL  ·  1.2s`, gray; otherwise `QWEN-VL READY`, gray

### Main loop
`parse_args()` (description = module docstring, `RawDescriptionHelpFormatter`):
`--vlm-model`, `--vlm-device` (env `VLM_DEVICE`, default `auto`),
`--max-new-tokens 60`, `--no-vlm`, `--detector`, `--detector-device` (env
`DETECTOR_DEVICE`, default `cpu`, which keeps the GPU free for Qwen-VL),
`--conf`, `--camera` (int, env `CAMERA_INDEX`, default 0), `--width 1280`,
`--height 720`, `--no-mirror`. Help strings mention the env var and default.

`open_camera(index, width, height)`: open it, set the size, and read up to 30
frames 50 ms apart until one is non-empty (some webcams return blank frames
while warming up). Return `None` on failure.

`main() -> int`:
1. Open the camera. On failure print a multi-line hint to stderr (another app
   using it? on macOS allow camera access under *System Settings > Privacy &
   Security > Camera*; try `--camera 1`) and return 1.
2. Load `YoloDetector` in a try/except. If it fails, print the error and
   carry on with `detector = None` (video + VLM still work).
3. Create `VlmWorker(QwenVL(...))` in a try/except (e.g. torch missing →
   keep `worker = None` and remember `short_error` for the footer). Unless
   `--no-vlm`, call `start_loading()` so the model loads while the window is
   already showing.
4. `cv2.namedWindow(..., WINDOW_AUTOSIZE)`, print a one-line key summary.
5. Each frame:
   1. Read. On a bad frame, count it; after 100 in a row print
      `Lost the webcam feed.` and stop. Sleep 20 ms, still honor `q`/Esc, and
      skip the frame.
   2. Mirror with `cv2.flip(frame, 1)` unless `--no-mirror`. Recreate the
      `Footer` if the frame width changed.
   3. Keep `clean_frame = frame`: this untouched image is what Qwen-VL gets,
      so it never sees the boxes.
   4. Run the detector (catch and print errors per frame). Update an
      exponentially smoothed FPS (`0.9 * old + 0.1 * new`).
   5. `view = frame.copy()`, draw boxes if enabled. Top-left pill:
      `OBJECT DETECTION  YOLO11N  30 FPS`, or `OBJECT DETECTION HIDDEN  (PRESS D)`,
      or `OBJECT DETECTION UNAVAILABLE`.
   6. For 0.25 s after a capture, blend white over the view, fading out, as a
      "frame captured" flash.
   7. Build the footer from `worker.snapshot()`: summary text is the
      unavailable reason, the error message, `"Qwen-VL is off. Press V to turn it on."`
      (when off and nothing described yet), or the latest summary. Show
      `np.vstack([view, footer])`.
   8. Keys: `q`/`Q`/Esc quit. `SPACE` → `worker.request(clean_frame)`; flash
      if accepted, else print that Qwen-VL is still busy (only one
      description at a time). `d` toggles boxes. `v` toggles Qwen-VL and
      starts loading it if it isn't loaded yet.
   9. Quit when the window is closed (`WND_PROP_VISIBLE < 1`).
6. Catch `KeyboardInterrupt`. In `finally`, release the camera,
   `destroyAllWindows()`, `waitKey(1)`. Return 0.

Under `if __name__ == "__main__":` call `main()`, flush stdout, then
`os._exit(code)` so the process exits immediately even if Qwen-VL is
mid-generation in its background thread.

## `README.md` specification

Write it for someone new to computer vision: first how to run it, then the
ideas, then the internals.

1. **Title and intro:** "Object Detection vs. Vision-Language Model". The two
   bullets (boxes = *what is there, and where?*; Qwen-VL sentence = *what is
   happening?*). Boxes come from the detector, the footer sentence from
   Qwen-VL, neither helps the other. Everything runs locally, no API keys.
2. **Setup:** Python 3.11+, a webcam, ~6 GB free disk. venv commands (with
   the Windows activate variant), `pip install -r requirements.txt`; note
   PyTorch is ~1 GB so it takes a few minutes.
3. **Run:** `python app.py`. A table of the two first-run downloads (YOLO11n
   ~5 MB → `yolo11n.pt` in the folder; Qwen3-VL-2B-Instruct ~4 GB → the
   Hugging Face cache). The window opens immediately and the footer says
   **LOADING QWEN-VL…** until it's ready. macOS camera-permission note.
4. **Controls:** "click the window first", key table (`SPACE`, `D`, `V`, `Q`),
   then what happens on `SPACE` as numbered steps (flash, background
   analysis, **ANALYZING…** with a timer while video keeps running, sentence +
   thumbnail replace the old one). Pressing `SPACE` while busy does nothing.
5. **What's on screen:** an ASCII mock-up of the window (pill top-left, two
   labeled boxes, footer with thumbnail, `VISION SUMMARY`, sentence,
   `QWEN-VL · 1.2s`) with arrows marking which part comes from which model,
   then a short legend.
6. **The ideas behind it**, numbered, plain-English, short paragraphs:
   1. What is computer vision? (a 1280×720 frame is 921,600 pixels × 3 numbers)
   2. What is object detection? (what + where; YOLO trained on COCO's 80
      classes; can't name anything else or describe actions)
   3. What does a bounding box represent? (four edges, a class, a confidence;
      the 40% threshold)
   4. How detection tells us what is present and where (an inventory:
      "Person. Laptop. Coffee cup." plus positions; a list, not an explanation)
   5. What is a vision-language model? (a language model that also takes an
      image; Qwen-VL from Alibaba's Qwen team)
   6. How Qwen-VL turns an image into a sentence (visual tokens from patches →
      joined with the prompt, quote it → written one word at a time; why it
      takes a second or two, why it's on demand, first-sentence-only)
   7. Object detection vs. visual understanding: a comparison table (output,
      question answered, gives you, vocabulary, speed) and a paragraph on why
      they complement each other (e.g. detector notices a change, VLM explains it).
7. **How the app works:** the two paths as small vertical diagrams (every
   frame vs. only on SPACE), the worker-thread explanation, the fact that the
   frame is copied before boxes are drawn, and a table mapping each section of
   `app.py` (`CONFIGURATION`, `YoloDetector`, `QwenVL`, `VlmWorker`,
   `draw_detections` / `Footer`, `main`) to what it does.
8. **Configuration:** table of setting / flag / env var / default for all
   eight user-facing settings, `python app.py --help`, three examples (4B
   model, `--conf 0.6`, `DETECTOR_MODEL=yolo11s.pt`). Then "Choosing a Qwen-VL
   model": a table of the three checkpoints with download size, memory
   needed (~6 / ~10 / ~9 GB) and notes; MPS on Apple Silicon, CUDA on NVIDIA,
   CPU fallback at 10–30 s per description; why the detector defaults to CPU.
9. **Troubleshooting** table: can't open webcam; **QWEN-VL ERROR** / could not
   load; not enough memory; inference failed; **OBJECT DETECTION
   UNAVAILABLE**; slow or choppy video; keys do nothing (click the window).
   Close with: if either model fails, the rest keeps running.
10. **Project files:** the file tree.

## Done when

- [ ] `pip install -r requirements.txt` works in a fresh venv
- [ ] `python app.py --help` prints the docstring and lists all eleven flags
- [ ] `python -c "import app"` works without a camera and without loading any
      model (heavy imports are lazy)
- [ ] `to_one_sentence` behaves: `'  "a man sits.  He reads."'` →
      `"A man sits."`, `"a dog"` → `"A dog."`, `""` → `""`
- [ ] `python app.py` opens the window right away with boxes and an FPS pill;
      the footer shows **LOADING QWEN-VL…**, then `QWEN-VL READY`. `SPACE`
      flashes, shows **ANALYZING…** while the video keeps moving, then a
      one-sentence description with a thumbnail. `D`, `V`, `Q`/Esc and
      closing the window all work. *(This is a manual check that needs a real
      camera and the ~4 GB model download. If you're an agent without a
      camera, say so instead of claiming it passed.)*
- [ ] `python app.py --vlm-model does/not-exist` keeps the video and boxes
      running and shows **QWEN-VL ERROR** with a "Could not load…" message
      in the footer
