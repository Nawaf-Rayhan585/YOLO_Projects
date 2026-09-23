# Rat / Rodent Detection

Detects rats/rodents in real-time footage — useful for warehouse, kitchen or
pest-control monitoring — and logs every sighting with a timestamp and
confidence score to a CSV file. Ships as a training notebook and a local
inference script.

<img width="829" alt="rat detection" src="https://github.com/user-attachments/assets/d2f78cb5-1811-404f-ae20-1d4b0ecbc6c8" />

## 1. Train — `train_rat_detector.ipynb`

Open in Colab (badge at the top), grab a rat/rodent dataset from
[Roboflow Universe](https://universe.roboflow.com), and run the cells top to
bottom on a free T4 GPU. Download `best.pt` at the end.

## 2. Run — `rat_detect.py`

```bash
pip install -r requirements.txt

python rat_detect.py --model best.pt --source warehouse.mp4
python rat_detect.py --model best.pt --source 0        # webcam
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `best.pt` | Path to your trained rodent-detection weights |
| `--source` | `0` | Video/image path or webcam index |
| `--conf` | `0.4` | Minimum detection confidence |
| `--log` | `sightings.csv` | CSV file every sighting is appended to |
| `--no-show` | off | Run headless, no preview window |

Press `q` to quit.

## Notes

- No pretrained rodent weights are bundled (dataset-specific) — train your
  own with the notebook, ~15-20 minutes on a free Colab T4.
- CPU is fine for running the trained model day-to-day; the GPU is only
  needed for training.
