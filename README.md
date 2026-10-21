# Object Detection vs. Vision-Language Model

A local webcam app that puts two kinds of computer vision side by side, on the
same image:

- **Object detection** draws boxes around things and names them: *what is there, and where is it?*
- **A vision-language model (Qwen-VL)** writes one sentence about the scene: *what is happening?*

The boxes on the video come from the object detector. The sentence in the black
footer comes from Qwen-VL. Neither model helps the other. Each one looks at the
image on its own.

Everything runs on your computer. No cloud APIs, no accounts, no API keys.

---

## Setup

You need **Python 3.11 or newer**, a webcam, and about **6 GB of free disk
space** for the models.

```bash
cd computer-vision-vlm
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`pip` installs PyTorch, which is large (around 1 GB), so this step can take a
few minutes.

## Run

```bash
python app.py
```

The first run downloads two models automatically:

| Model | Size | Saved to |
| --- | --- | --- |
| YOLO11n (object detection) | ~5 MB | `yolo11n.pt` in this folder |
| Qwen3-VL-2B-Instruct (vision-language) | ~4 GB | the Hugging Face cache (`~/.cache/huggingface`) |

The window opens right away and the webcam and object detection start
immediately. The footer says **LOADING QWEN-VL…** until the language model is
ready, which takes a few seconds once it has been downloaded.

**macOS note:** the first run may ask for camera permission. If the app says it
cannot open the webcam, allow your terminal app (Terminal, iTerm, VS Code, ...)
in **System Settings → Privacy & Security → Camera**, then run it again.

## Controls

Click the window first so it receives your key presses.

| Key | What happens |
| --- | --- |
| `SPACE` | Capture the current frame and ask Qwen-VL to describe it |
| `D` | Show or hide the object-detection boxes |
| `V` | Turn Qwen-VL on or off |
| `Q` | Quit (`Esc` or closing the window also works) |

When you press `SPACE`:

1. The screen flashes to show that a frame was captured.
2. That single frame goes to Qwen-VL in the background.
3. The footer shows **ANALYZING…** with a timer. The video and boxes keep running.
4. The new sentence replaces the old one, and a small thumbnail shows the
   exact frame it describes.

Pressing `SPACE` again while Qwen-VL is still working does nothing; only one
description runs at a time.

## What's on screen

```
┌──────────────────────────────────────────────────────────────┐
│ OBJECT DETECTION  YOLO11N  30 FPS                            │
│                                                              │
│        ┌ PERSON 92% ──────────┐      ┌ LAPTOP 88% ─┐         │  ← object detector
│        │                      │      │             │         │
│        └──────────────────────┘      └─────────────┘         │
├──────────────────────────────────────────────────────────────┤
│ [thumb]  VISION SUMMARY                     QWEN-VL · 1.2s   │  ← Qwen-VL
│          A person is sitting at a desk working on a laptop.  │
└──────────────────────────────────────────────────────────────┘
```

- **Top-left:** which detector is running and how many frames per second it handles.
- **Boxes:** one per detected object, labelled with its class and confidence.
- **Footer:** the latest Qwen-VL sentence, its status, and how long it took.

---

## The ideas behind it

### 1. What is computer vision?

To a computer, a photo is a grid of numbers. A 1280 × 720 webcam frame is
921,600 pixels, and each pixel is three numbers for how much blue, green, and
red it contains. **Computer vision** is the field of getting useful information
out of those numbers: what objects are in the picture, where they are, what is
going on.

### 2. What is object detection?

**Object detection** answers two questions at once:

- **What** objects are in the image? (a person, a laptop, a cup)
- **Where** is each one?

The detector in this app is **YOLO** ("You Only Look Once"). It was trained on
a dataset called COCO, so it knows 80 everyday categories: person, cup, laptop,
cell phone, chair, dog, bottle, book, and so on. It is small and fast enough to
run on every webcam frame.

The detector can only name things from its list of 80 categories. It has no
words for anything else, and it doesn't describe what anyone is doing.

### 3. What does a bounding box represent?

A **bounding box** is the smallest rectangle that contains one object. The
detector reports each box as four numbers: the left, top, right, and bottom
edges in pixels. Along with the box it gives:

- a **class**: which of its known categories it thinks the object is (`PERSON`)
- a **confidence**: how sure it is, from 0% to 100% (`92%`)

Boxes below the confidence threshold (40% by default) are hidden. Raise the
threshold to see fewer, surer boxes. Lower it to see more guesses.

### 4. How object detection tells us what is present and where

Put the boxes together and you get an inventory of the scene:

> **Person. Laptop. Coffee cup.**

…plus each item's position. You can count the people, check whether a cup
sits inside the area of a desk, or follow an object as it moves from frame to
frame. That is very useful, but it is a list of items, not an explanation of
what is going on.

### 5. What is a vision-language model?

A **vision-language model (VLM)** is a language model, like the ones behind
chat assistants, that can also take an image as input. You give it a picture and
a question in plain English, and it answers in plain English.

This app uses **Qwen-VL**, an open model family from Alibaba's Qwen team that
you can download and run on your own computer.

### 6. How Qwen-VL turns an image into a sentence

1. **The image becomes "visual tokens."** A vision encoder cuts the image into
   small patches and turns each patch into a list of numbers that captures what
   it contains: an edge, a texture, part of a face.
2. **The tokens join the prompt.** Those visual tokens are placed in front of
   the text instruction. This app's instruction is:
   *"Describe what is happening in this image in exactly one concise sentence…
   Do not speculate about anything that cannot be seen."*
3. **The language model writes the answer one word at a time.** It looks at the
   picture and the instruction together, which lets it relate things to each
   other: who is holding what, what is next to what, what someone seems to be
   doing.

Because this is far more work than drawing boxes, it takes a second or two per
image instead of a few milliseconds. That's why the app only runs it when you
press `SPACE`.

The app also makes sure the result is a single sentence. If the model returns
more than one, only the first is kept.

### 7. Object detection vs. visual understanding

The same webcam frame, two different answers:

| | Object detection | Vision-language model |
| --- | --- | --- |
| Output | **"Person. Laptop. Coffee cup."** | **"A person is sitting at a desk working on a laptop with a coffee cup nearby."** |
| Question it answers | What is there, and where is it? | What is happening in the scene? |
| Gives you | Labels, exact boxes, confidence scores | Actions, relationships, overall context |
| Vocabulary | A fixed list of 80 classes | Any words |
| Speed | Every frame (milliseconds) | On demand (about a second or two) |

**They complement each other.** The detector is fast and precise about
*where* things are, which is great for counting, tracking, and triggering
actions. The VLM is slower, but it understands *how things relate*, which is
what you want for describing or summarizing a scene. Real systems often use
both, for example a detector to notice that something changed and a VLM to
explain what it was.

---

## How the app works

Two separate paths read from the same webcam.

**Path 1 runs on every frame:**

```
WEBCAM
  ↓
OBJECT DETECTION (YOLO)
  ↓
BOUNDING BOXES  →  drawn on the video
```

**Path 2 runs independently, only when you press SPACE:**

```
WEBCAM FRAME (captured on SPACE)
  ↓
QWEN-VL (background thread)
  ↓
VISION SUMMARY  →  written in the black footer
```

The main loop reads the webcam, runs the detector, draws the boxes and footer,
and checks for key presses, roughly 30 times a second. Qwen-VL runs in a
separate **worker thread**, so the main loop never waits for it. The frame is
copied before any boxes are drawn, which means Qwen-VL sees the plain image and
no detector output.

Everything is in `app.py`, organised top to bottom:

| Section | What it does |
| --- | --- |
| `CONFIGURATION` | Model names, confidence threshold, the Qwen-VL prompt |
| `YoloDetector` | Wraps YOLO. To use a different detector, write a class with the same `detect(frame)` method. |
| `QwenVL` | Loads Qwen-VL and turns one image into one sentence |
| `VlmWorker` | Runs Qwen-VL in the background, one request at a time |
| `draw_detections`, `Footer` | Draw the boxes and the black footer |
| `main` | The webcam loop and keyboard controls |

---

## Configuration

Change settings with command-line flags, with environment variables, or by
editing the `CONFIGURATION` section at the top of `app.py`.

| Setting | Flag | Environment variable | Default |
| --- | --- | --- | --- |
| Qwen-VL model | `--vlm-model` | `QWEN_VL_MODEL` | `Qwen/Qwen3-VL-2B-Instruct` |
| Qwen-VL device | `--vlm-device` | `VLM_DEVICE` | `auto` (CUDA → Apple MPS → CPU) |
| Detector weights | `--detector` | `DETECTOR_MODEL` | `yolo11n.pt` |
| Detector device | `--detector-device` | `DETECTOR_DEVICE` | `cpu` |
| Confidence threshold | `--conf` | `DETECTION_CONFIDENCE` | `0.4` |
| Webcam | `--camera` | `CAMERA_INDEX` | `0` |
| Start with Qwen-VL off | `--no-vlm` | | off |
| Don't mirror the video | `--no-mirror` | | mirrored |

Run `python app.py --help` to see every option.

Examples:

```bash
# A bigger, more detailed Qwen model (about 8 GB, slower)
python app.py --vlm-model Qwen/Qwen3-VL-4B-Instruct

# Show only detections the model is at least 60% sure about
python app.py --conf 0.6

# A larger, more accurate YOLO model
DETECTOR_MODEL=yolo11s.pt python app.py
```

### Choosing a Qwen-VL model

| Model | Download | Memory needed | Notes |
| --- | --- | --- | --- |
| `Qwen/Qwen3-VL-2B-Instruct` | ~4 GB | ~6 GB | **Default.** About 1 second per description on an Apple M-series Mac. |
| `Qwen/Qwen3-VL-4B-Instruct` | ~8 GB | ~10 GB | More detail, roughly twice as slow |
| `Qwen/Qwen2.5-VL-3B-Instruct` | ~7 GB | ~9 GB | Previous generation |

On an Apple Silicon Mac the model runs on the GPU (MPS). On a PC with an NVIDIA
GPU it uses CUDA. Without a GPU it falls back to the CPU, which works but can
take 10–30 seconds per description.

The detector runs on the CPU by default. YOLO11n handles 60+ FPS there, and
keeping it off the GPU leaves the GPU free for Qwen-VL. On a slower CPU, try
`--detector-device mps` (Mac) or `--detector-device cuda` (NVIDIA).

---

## Troubleshooting

| Problem | What to do |
| --- | --- |
| "Could not open webcam" | Close other apps using the camera, allow camera access (macOS note above), or try `--camera 1`. |
| Footer says **QWEN-VL ERROR** / "Could not load…" | Check the terminal for the full error. Usually the model name is wrong or the download was interrupted (check your internet connection for the first run). Press `SPACE` to retry. |
| "Not enough memory for this model" | Close other apps, or use a smaller model such as the 2B default. |
| "Inference failed. Press SPACE to try again." | One description failed; the app keeps running. Press `SPACE` again. |
| Top-left says **OBJECT DETECTION UNAVAILABLE** | The YOLO weights could not be loaded or downloaded. Check the terminal and your internet connection. The webcam and Qwen-VL keep working. |
| Video is slow or choppy | Use the default `yolo11n.pt`, or try `--detector-device mps` / `cuda`. |
| Keys do nothing | Click the video window so it has keyboard focus. |

If either model fails, the rest of the app keeps running. A Qwen-VL problem
never stops the webcam or the boxes.

## Project files

```
computer-vision-vlm/
├── app.py            the whole application
├── requirements.txt  Python dependencies
├── README.md         this file
└── .gitignore
```
