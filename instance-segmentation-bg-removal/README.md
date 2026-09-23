# Instance Segmentation Background Removal

A green-screen effect with no green screen: uses a YOLO **segmentation**
model (not just bounding boxes) to cut a person's exact silhouette out of
the frame in real time, then composites them over a blurred background, a
solid color, or any image you supply.

## How it works

1. A YOLO segmentation checkpoint (`yolo11n-seg.pt`) produces a pixel-level
   mask for each detected instance of the target class (person, by default).
2. All instance masks are merged into one binary mask.
3. `np.where` composites the original frame's foreground pixels (where the
   mask is set) over a chosen background — a Gaussian-blurred version of the
   original frame, a flat color, or a static image resized to match.

## Usage

```bash
pip install -r requirements.txt

python background_removal.py --source 0 --mode blur
python background_removal.py --source 0 --mode color --color 0,255,0
python background_removal.py --source call.mp4 --mode image --bg office.jpg
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Video file path or webcam index |
| `--model` | `yolo11n-seg.pt` | Ultralytics segmentation checkpoint |
| `--conf` | `0.5` | Minimum detection confidence |
| `--class-id` | `0` | COCO class kept in the foreground (0 = person) |
| `--mode` | `blur` | `blur`, `color`, or `image` |
| `--color` | `0,255,0` | BGR background color for `--mode color` |
| `--bg` | `None` | Background image path for `--mode image` |
| `--no-show` | off | Run headless, no preview window |

Press `q` to quit.

## Notes

- `--class-id` isn't limited to people — point it at any COCO class the
  segmentation model supports (e.g. `16` for dogs) to cut out something else
  entirely.
- Mask edges are as clean as the underlying segmentation model; swap in
  `yolo11s-seg.pt` / `yolo11m-seg.pt` for crisper edges at the cost of speed.
