# Crowd Heatmap

Generates a live heatmap of crowd density and movement — the kind of
overlay used for retail foot-traffic analysis or venue capacity monitoring —
while also drawing individual detections and a live person count.

<img width="873" alt="crowd heatmap" src="https://github.com/user-attachments/assets/52b49e33-c02b-477a-a100-18216917b883" />

## How it works

1. YOLO detects people in each frame.
2. Each detection's center point adds a "heat" blob to an accumulator the
   same size as the frame.
3. The accumulator decays slightly every frame (`--decay`) so old activity
   fades out rather than permanently saturating one spot, then gets
   Gaussian-blurred, normalized, and colorized with `COLORMAP_JET`.
4. The colorized heat is blended back over the live video.

## Usage

```bash
pip install -r requirements.txt

python crowd_heatmap.py --source people.mp4
python crowd_heatmap.py --source 0 --decay 0.95 --radius 40
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Video file path or webcam index |
| `--model` | `yolo11n.pt` | Any Ultralytics YOLO checkpoint |
| `--conf` | `0.4` | Minimum detection confidence |
| `--decay` | `0.98` | How fast old heat fades (closer to 1 = slower fade) |
| `--radius` | `30` | Pixel radius of heat added per detection |
| `--no-show` | off | Run headless, no preview window |

Press `Esc` to quit.
