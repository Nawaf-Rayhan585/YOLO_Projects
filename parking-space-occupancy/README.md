# Parking Space Occupancy

Marks each parking space in a fixed camera view as **occupied** or **free**
in real time, with a live "Free: X / Total: Y" counter — built for a static
overhead or angled parking-lot camera. Two-step workflow: click out the
slots once, then run detection on any footage from that same camera.

## How it works

1. **`define_parking_slots.py`** — click the corners of each parking space
   on a reference frame (an image, or the first frame of a video). Press
   `n` after each slot to start the next one, `s`/`q` to save. Slots are
   saved as a list of polygons in a JSON file.
2. **`parking_occupancy.py`** — runs YOLO vehicle detection on the video,
   and for every slot polygon checks whether any detected vehicle's center
   point falls inside it (`cv2.pointPolygonTest`). Occupied slots are
   shaded red, free ones green.

## Usage

```bash
pip install -r requirements.txt

# 1. define your slots once (on an image or a video's first frame)
python define_parking_slots.py --video lot.mp4 --out slots.json

# 2. run occupancy detection using those slots
python parking_occupancy.py --source lot.mp4 --slots slots.json
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Video file path or webcam index |
| `--slots` | `slots.json` | Slot polygons produced by `define_parking_slots.py` |
| `--model` | `yolo11n.pt` | Any Ultralytics YOLO checkpoint |
| `--conf` | `0.35` | Minimum detection confidence |
| `--no-show` | off | Run headless, no preview window |

Press `q` to quit.

## Notes

- Slots are tied to a specific camera framing — if the camera moves, redefine
  them.
- Works best with a camera angle where parked vehicles don't heavily overlap
  each other in the 2D projection.
