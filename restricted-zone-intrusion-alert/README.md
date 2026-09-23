# Restricted Zone Intrusion Alert

Watches a defined polygon area in a camera feed — a "staff only" section,
a machine safety perimeter, a fenced-off zone — and raises a visible
on-screen alert plus an audible beep the moment a person's feet land
inside it.

<img width="406" alt="human alert" src="https://github.com/user-attachments/assets/aec7794c-c44f-4b07-9ede-d12e709b8088" />

## How it works

1. YOLO detects people in each frame.
2. Each detection's foot point (bottom-center of its bounding box) is tested
   against the zone polygon with `cv2.pointPolygonTest`, which is far more
   accurate for ground-plane intrusion than testing the whole box.
3. If any foot point falls inside the zone, the zone is highlighted red, an
   "INTRUSION DETECTED" banner is drawn, and a beep fires (rate-limited by
   `--cooldown` so it doesn't spam).

## Usage

```bash
pip install -r requirements.txt

# default zone: a rectangle covering the middle of the frame
python intrusion_alert.py --source cctv.mp4

# custom zone polygon, in pixel coordinates, drawn clockwise or counter-clockwise
python intrusion_alert.py --source cctv.mp4 --zone 100,100 500,100 500,400 100,400
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Video file path or webcam index |
| `--model` | `yolo11n.pt` | Any Ultralytics YOLO checkpoint |
| `--conf` | `0.5` | Minimum detection confidence |
| `--zone` | mid-frame rectangle | Polygon points as `x,y x,y ...` |
| `--cooldown` | `3.0` | Minimum seconds between repeated alerts |
| `--mute` | off | Disable the audio beep |
| `--no-show` | off | Run headless, no preview window |

## Notes

- To find good zone coordinates for your own camera, pause on a frame in any
  image viewer and read off pixel positions of the corners you want.
- The beep uses `winsound` on Windows and the terminal bell elsewhere —
  swap in your own siren/SMS/webhook call inside `beep()` for production use.
