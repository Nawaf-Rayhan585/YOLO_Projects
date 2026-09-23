# Fall Detection

Flags a person as fallen when their body center **drops sharply** and their
posture goes from upright to **horizontal** at the same time — the kind of
monitoring used for elderly care or lone-worker safety. Built entirely on
YOLO pose estimation and per-person tracking; no training data needed.

> Heuristic demo, not a certified medical/safety device — tune the
> thresholds for your camera height/angle and treat this as a proof of
> concept, not a deployed safety system.

## How it works

1. `YOLO(...).track()` (pose variant) tracks each person and their 17 COCO
   body keypoints frame to frame.
2. A "body center" is computed as the midpoint between the shoulder and hip
   keypoints, and its vertical position is kept in a short rolling window
   (`--window` frames).
3. **Drop signal:** if the body center moves down by more than
   `--drop-thresh` (as a fraction of the person's own bounding-box height)
   within that window, that's a sudden downward motion.
4. **Lying signal:** if the bounding box becomes wider than it is tall
   (`--lying-ratio`), the posture looks horizontal rather than standing/
   sitting.
5. Both together → flagged as fallen (red box, on-screen banner, beep,
   rate-limited per person). The flag clears once the person's box goes back
   to a non-horizontal aspect ratio (e.g. they get back up).

## Usage

```bash
pip install -r requirements.txt

python fall_detector.py --source room.mp4
python fall_detector.py --source 0
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Video file path or webcam index |
| `--model` | `yolo11n-pose.pt` | Ultralytics pose checkpoint |
| `--conf` | `0.5` | Minimum detection confidence |
| `--window` | `8` | Frames looked back over for the drop measurement |
| `--drop-thresh` | `0.5` | Vertical drop (fraction of bbox height) that counts as a fall trigger |
| `--lying-ratio` | `1.2` | Bbox width/height above which posture counts as lying down |
| `--cooldown` | `5.0` | Minimum seconds between repeated alerts per person |
| `--mute` | off | Disable the audio alarm |
| `--no-show` | off | Run headless, no preview window |

Press `q` to quit.
