# License Plate Recognition (ANPR)

Automatic Number Plate Recognition: a YOLO model locates license plates in
a video feed, then [EasyOCR](https://github.com/JaidedAI/EasyOCR) reads the
characters off each detected plate. Every unique plate read is logged with a
timestamp to a CSV file — useful for parking access control, toll systems,
or gate logging.

## How it works

1. `train_plate_detector.ipynb` trains a lightweight YOLO model to do one
   job: draw a box around license plates (it doesn't need to know what the
   text says).
2. `anpr_inference.py` runs that detector on each frame, crops out every
   detected plate region, and passes the crop to EasyOCR.
3. Plate text is cleaned (letters + digits only, uppercased) and deduplicated
   by rough screen position so the same plate isn't logged every frame while
   parked in view.

## 1. Train — `train_plate_detector.ipynb`

Open in Colab, grab a license-plate dataset from
[Roboflow Universe](https://universe.roboflow.com), train on the free T4
GPU, and download `best.pt`.

## 2. Run — `anpr_inference.py`

```bash
pip install -r requirements.txt

python anpr_inference.py --model best.pt --source parking_lot.mp4
python anpr_inference.py --model best.pt --source 0
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `best.pt` | Path to your trained plate-detection weights |
| `--source` | `0` | Video/image path or webcam index |
| `--conf` | `0.4` | Detection confidence threshold |
| `--ocr-conf` | `0.4` | Minimum OCR confidence to accept a reading |
| `--log` | `plates.csv` | CSV file recognized plates are appended to |
| `--gpu` | off | Use GPU for EasyOCR if you have one |
| `--no-show` | off | Run headless, no preview window |

Press `q` to quit.

## Notes

- EasyOCR downloads its recognition model on first run (a few hundred MB).
- OCR accuracy depends heavily on plate resolution — a plate under ~60px
  wide in the source frame will read poorly regardless of the model.
