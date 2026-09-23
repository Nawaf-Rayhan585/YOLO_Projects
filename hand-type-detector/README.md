# Hand Type Detector (Left / Right)

Detects every hand in frame and labels it **Left** or **Right**, using
[cvzone](https://github.com/cvzone/cvzone)'s MediaPipe-based hand tracker.
A simple, fast building block for gesture-control or sign-language projects
that need to know which hand is doing what.

<img width="332" alt="hand type detection" src="https://github.com/user-attachments/assets/f3caf1ee-1f35-42e9-9856-6dbe0fcb5d9d" />

## How it works

MediaPipe's hand landmark model reports handedness per detected hand, but
that label is mirrored on a front-facing webcam. This script falls back to
a position-based check (hand center relative to frame midpoint) if MediaPipe
doesn't return a confident Left/Right label, and color-codes each label
(green = right, orange = left) directly over the hand.

## Usage

```bash
pip install -r requirements.txt

python hand_type_detector.py --source 0          # webcam
python hand_type_detector.py --source hands.mp4   # video file
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Video file path or webcam index |
| `--max-hands` | `2` | Maximum simultaneous hands to track |
| `--conf` | `0.8` | Detection confidence threshold |
| `--no-show` | off | Run headless, no preview window |

Press `q` to quit.
