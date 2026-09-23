# Vehicle Detection & Tracking

Detects cars, motorcycles, buses and trucks in a video stream and tracks each
one with a persistent ID using YOLO + ByteTrack, so the same vehicle keeps
the same number as it moves through the frame instead of being re-counted
every frame.

<img width="410" alt="vehicle tracking" src="https://github.com/user-attachments/assets/7159c697-57a4-4b60-812b-567bad135ccd" />

## How it works

1. Each frame is run through a YOLO model, filtered to the four COCO vehicle
   classes (`car`, `motorcycle`, `bus`, `truck`).
2. Detections are handed to [ByteTrack](https://github.com/ifzhang/ByteTrack)
   (via the `supervision` library) which assigns and maintains a track ID per
   vehicle across frames.
3. A running per-class count of unique track IDs seen is kept and drawn live
   on the frame, plus printed as a summary when the video ends.

## Usage

```bash
pip install -r requirements.txt

# on a video file
python vehicle_tracker.py --source cars.mp4

# on a webcam
python vehicle_tracker.py --source 0

# save the annotated output
python vehicle_tracker.py --source cars.mp4 --save output.mp4
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Video file path or webcam index |
| `--model` | `yolo11n.pt` | Any Ultralytics YOLO checkpoint (auto-downloads) |
| `--conf` | `0.4` | Minimum detection confidence |
| `--save` | `None` | Path to write the annotated video to |
| `--no-show` | off | Run headless, no preview window |

Press `q` to quit the preview window.

## Notes

- No video is bundled with this project (keeps the repo small) — point
  `--source` at your own clip, or `0` for a live webcam.
- Swap `--model` for `yolo11s.pt` / `yolo11m.pt` for higher accuracy at the
  cost of speed.
