# Suspicious Behavior Detection

Flags a person as "suspicious" when they are **bent/crouching** and **moving
their hands quickly** at the same time — a simple, explainable proxy for
loss-prevention style behavior (reaching into a shelf, pocketing an item)
that runs live off pose estimation with no training required.

<img width="332" alt="suspicious detection" src="https://github.com/user-attachments/assets/f3caf1ee-1f35-42e9-9856-6dbe0fcb5d9d" />

> **This is a heuristic demo, not a trained action classifier.** It's meant
> to show how far you can get combining pose keypoints with simple rules,
> and as a starting point for collecting labeled clips to eventually train a
> real action-recognition model on. Tune the thresholds for your own camera
> angle before trusting it for anything real.

## How it works

1. `YOLO(...).track()` (pose variant) detects people, estimates 17 COCO body
   keypoints per person, and assigns each a persistent track ID frame to
   frame.
2. **Crouch signal:** a person's bounding box gets wider relative to its
   height when they bend over — flagged when `width / height` exceeds
   `--crouch-ratio`.
3. **Fast-hands signal:** the midpoint of both wrist keypoints is compared to
   its position last frame; movement above `--move-thresh` (normalized by
   the box diagonal, so it's resolution-independent) counts as "fast".
4. When both signals hold for `--sustain-frames` consecutive frames, that
   track is flagged and boxed in red instead of green.

## Usage

```bash
pip install -r requirements.txt

python suspicious_behavior.py --source store.mp4
python suspicious_behavior.py --source 0 --crouch-ratio 0.9 --sustain-frames 5
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Video file path or webcam index |
| `--model` | `yolo11n-pose.pt` | Ultralytics pose checkpoint |
| `--conf` | `0.5` | Minimum detection confidence |
| `--crouch-ratio` | `0.85` | Bbox width/height above which a person counts as bent over |
| `--move-thresh` | `0.06` | Per-frame wrist movement (fraction of bbox diagonal) counted as fast |
| `--sustain-frames` | `8` | Consecutive suspicious frames needed to flag |
| `--no-show` | off | Run headless, no preview window |

Press `q` to quit.
