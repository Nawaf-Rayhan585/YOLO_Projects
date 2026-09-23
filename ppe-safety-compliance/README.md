# PPE Safety Compliance Detection

Flags workers missing required safety gear — hard hats, safety vests, masks —
on a construction site or factory floor feed. Detects both the gear and its
absence, so a "NO-Hardhat" detection draws a red box and gets logged as a
violation while a "Hardhat" detection draws green.

## How it works

`ppe_compliance.py` doesn't hardcode a class list — it treats **any**
detected class whose name contains `no-`, `no_`, `without` or `missing` as a
violation. This means most public PPE/construction-safety datasets (which
almost universally label both the presence and the absence of each item)
work out of the box with zero code changes, only a different trained model.

## 1. Train — `train_ppe_detector.ipynb`

Open in Colab, grab a PPE/construction-site-safety dataset from
[Roboflow Universe](https://universe.roboflow.com), train on the free T4
GPU, download `best.pt`.

## 2. Run — `ppe_compliance.py`

```bash
pip install -r requirements.txt

python ppe_compliance.py --model best.pt --source site.mp4
python ppe_compliance.py --model best.pt --source 0
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `best.pt` | Path to your trained PPE-detection weights |
| `--source` | `0` | Video/image path or webcam index |
| `--conf` | `0.4` | Detection confidence threshold |
| `--cooldown` | `5.0` | Minimum seconds between repeated violation logs |
| `--log` | `violations.csv` | CSV file violations are appended to |
| `--no-show` | off | Run headless, no preview window |

Press `q` to quit. A per-class summary prints when the video ends.

## Notes

- No pretrained PPE weights are bundled (dataset-specific) — train your own
  with the notebook above.
- Works for any equipment your dataset labels this way, not just hard hats —
  gloves, goggles, harnesses, hi-vis vests, etc.
