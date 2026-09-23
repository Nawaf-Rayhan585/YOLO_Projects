# Computer Vision Projects

Made with ❤️ by **Nawaf Rayhan**

⭐ If this repo helped you, please give it a star!

A collection of 19 standalone, runnable computer vision projects — YOLO
detection & tracking, pose estimation, segmentation, MediaPipe, and classical
OpenCV — covering surveillance, retail analytics, safety, productivity, and
document processing. Every project lives in its own folder with its own
script(s), `README.md`, and `requirements.txt`, so you can clone the whole
repo or just grab the one folder you need.

---

## 📌 Projects

### Surveillance & Safety

| Project | What it does |
|---|---|
| [restricted-zone-intrusion-alert](restricted-zone-intrusion-alert) | Alerts when a person's feet enter a defined restricted zone |
| [suspicious-behavior-detection](suspicious-behavior-detection) | Pose-based heuristic: flags crouching + fast hand movement |
| [fire-smoke-detection](fire-smoke-detection) | Real-time fire detection with alarm, + training notebook |
| [fall-detection](fall-detection) | Pose-based heuristic: flags a sudden drop into a horizontal posture |
| [ppe-safety-compliance](ppe-safety-compliance) | Flags missing hard hats / safety vests / masks on site |
| [drowsiness-yawn-detection](drowsiness-yawn-detection) | Driver-monitoring style eye-closure + yawn detection |
| [face-blur-anonymizer](face-blur-anonymizer) | Blurs/pixelates every face in a feed for privacy compliance |
| [rat-rodent-detection](rat-rodent-detection) | Rodent detection + sighting log, + training notebook |

### Retail & Crowd Analytics

| Project | What it does |
|---|---|
| [people-counter-in-out](people-counter-in-out) | IN/OUT footfall counter with a live dashboard |
| [crowd-heatmap](crowd-heatmap) | Live crowd density heatmap + person count |
| [queue-wait-time-estimator](queue-wait-time-estimator) | Live queue length + rolling average wait time |
| [parking-space-occupancy](parking-space-occupancy) | Per-slot free/occupied detection for a parking lot |

### Traffic & Vehicles

| Project | What it does |
|---|---|
| [vehicle-detection-tracking](vehicle-detection-tracking) | Multi-class vehicle detection with persistent tracking IDs |
| [license-plate-recognition](license-plate-recognition) | ANPR: plate detection + OCR reading, + training notebook |

### Productivity & HCI

| Project | What it does |
|---|---|
| [focus-guard-distraction-detector](focus-guard-distraction-detector) | Self-monitoring screen-distraction guard with full-screen alerts |
| [hand-type-detector](hand-type-detector) | Labels each detected hand Left / Right |
| [hand-gesture-volume-control](hand-gesture-volume-control) | Pinch-gesture system volume control |

### Document & Vision Utilities

| Project | What it does |
|---|---|
| [document-scanner-ocr](document-scanner-ocr) | Classic CV document edge-detect + perspective warp + OCR |
| [instance-segmentation-bg-removal](instance-segmentation-bg-removal) | Green-screen-style background blur/replace via segmentation |

---

## 🛠️ Tech Stack

- **Detection & tracking:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (v8/v11) + [ByteTrack](https://github.com/ifzhang/ByteTrack) via `supervision`
- **Pose & hands:** YOLO-Pose, [MediaPipe](https://google.github.io/mediapipe/), [cvzone](https://github.com/cvzone/cvzone)
- **Segmentation:** YOLO-Seg
- **OCR:** [EasyOCR](https://github.com/JaidedAI/EasyOCR), [Tesseract](https://github.com/tesseract-ocr/tesseract)
- **Core:** Python, OpenCV, NumPy
- **Training:** Google Colab notebooks + [Roboflow](https://roboflow.com) datasets, for the projects that need a custom-trained model

## ⚙️ Installation

Each project is self-contained — clone the repo, then install only what
that project needs:

```bash
git clone https://github.com/Nawaf-Rayhan585/YOLO_Projects.git
cd YOLO_Projects/<project-folder>
pip install -r requirements.txt
```

## ▶️ Usage

Every script takes a `--source` argument (a video file path, an image path,
or a webcam index like `0`) plus project-specific flags — see each folder's
own `README.md` for the exact command and options. General pattern:

```bash
python <script>.py --source your_video.mp4
python <script>.py --source 0              # webcam
```

## 📁 Project Structure

Two kinds of projects in this repo:

- **Ready to run** — most projects use a general-purpose pretrained YOLO/
  MediaPipe model and work immediately after `pip install`.
- **Train then run** — a few (fire, rodent, license plate, PPE detection)
  need a model trained on a domain-specific dataset. Those folders include a
  Colab notebook (`train_*.ipynb`) that trains one on a free GPU in
  ~15-30 minutes, plus a local `*_inference.py` / `*_detect.py` script to run
  the result.

No video/image datasets or trained weights are committed to the repo (keeps
it light) — point each script's `--source`/`--model` flags at your own.

## 📜 License

[MIT](LICENSE) — free to use, modify, and build on.

## 📬 Contact

Built by **Nawaf Rayhan** — feel free to reach out for freelance CV/AI work.

⭐ Don't forget to star the repo if you found it useful!
