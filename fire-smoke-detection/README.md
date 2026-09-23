# Fire Detection

Detects fire in real time on video, webcam, or still images and raises an
on-screen + audible alarm the moment it's spotted. Ships as two pieces:
a Colab notebook to train the detector, and a local script to run it.

<img width="459" height="260" alt="fire detection" src="https://github.com/user-attachments/assets/9c28b0f5-6286-413c-a42d-ae5436991f93" />

## 1. Train — `train_fire_detector.ipynb`

Open the notebook in Colab (badge at the top), point it at a fire dataset
from [Roboflow Universe](https://universe.roboflow.com) (or your own, in
YOLO format), and run the cells top to bottom. Training uses a free T4 GPU;
running the resulting model afterwards does not need a GPU at all. At the
end, download `best.pt`.

## 2. Run — `fire_detect.py`

```bash
pip install -r requirements.txt

python fire_detect.py --model best.pt --source warehouse.mp4
python fire_detect.py --model best.pt --source 0        # webcam
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `best.pt` | Path to your trained fire-detection weights |
| `--source` | `0` | Video/image path or webcam index |
| `--conf` | `0.4` | Minimum detection confidence |
| `--cooldown` | `5.0` | Minimum seconds between repeated alarms |
| `--mute` | off | Disable the audio alarm |
| `--no-show` | off | Run headless, no preview window |

Press `q` to quit.

## Notes

- No pretrained fire-detection weights are bundled here (they're
  dataset-specific) — train your own with the notebook above, it only
  takes ~20-30 minutes on a free Colab T4.
- Swap `yolov8s.pt` for `yolov8n.pt` in the notebook if you want a smaller,
  faster model for edge deployment.
