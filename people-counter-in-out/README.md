# People Counter (IN / OUT)

Counts people entering and exiting through a virtual line in real time —
the classic retail/venue footfall counter. Each person is tracked with a
persistent ID so they're only counted once per crossing, not once per frame.

<img width="404" alt="people counting" src="https://github.com/user-attachments/assets/7bd05e4d-3f74-43fb-ad43-689e0317cd33" />

## How it works

1. YOLO detects people (COCO class `0`) in every frame.
2. ByteTrack assigns a stable ID to each person across frames.
3. Each ID's vertical position is compared frame-to-frame against a
   configurable line — crossing downward increments **IN**, crossing upward
   increments **OUT**.
4. A dashboard overlay shows the live video feed plus running totals.

## Usage

```bash
pip install -r requirements.txt

python people_counter.py --source people.mp4
python people_counter.py --source 0 --line 240   # webcam, custom line height
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Video file path or webcam index |
| `--model` | `yolo11n.pt` | Any Ultralytics YOLO checkpoint |
| `--conf` | `0.4` | Minimum detection confidence |
| `--line` | mid-frame | Y-coordinate of the counting line |
| `--no-show` | off | Run headless, no preview window |

Press `Esc` to quit the preview window.

## Notes

- Best results with a mostly-overhead or angled camera and a line placed
  across a doorway/gate rather than a busy open area.
- No sample video is bundled — supply your own via `--source`.
