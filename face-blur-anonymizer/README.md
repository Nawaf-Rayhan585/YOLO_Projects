# Face Blur Anonymizer

Detects every face in a video or webcam feed and blurs, pixelates, or blacks
it out in real time — for publishing security footage, meeting GDPR/privacy
requirements, or anonymizing bystanders before sharing a clip. Runs fully
offline with OpenCV's bundled Haar cascade face detector — **no model
download and no training required**, works the moment you `pip install`.

## How it works

1. Each frame is converted to grayscale and run through OpenCV's
   `haarcascade_frontalface_default` detector.
2. Every detected face region is replaced in-place with either a strong
   Gaussian blur, a pixelated (mosaic) version, or a solid black box.
3. The rest of the frame is untouched.

## Usage

```bash
pip install -r requirements.txt

python face_anonymizer.py --source crowd.mp4
python face_anonymizer.py --source 0 --mode pixelate
python face_anonymizer.py --source crowd.mp4 --mode box --save anonymized.mp4
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Video file path or webcam index |
| `--mode` | `blur` | `blur`, `pixelate`, or `box` |
| `--scale-factor` | `1.1` | Haar cascade scale factor (lower = more thorough, slower) |
| `--min-neighbors` | `6` | Higher = fewer false-positive face boxes |
| `--save` | `None` | Path to write the anonymized video to |
| `--no-show` | off | Run headless, no preview window |

Press `q` to quit.

## Notes

- Haar cascades are fast and dependency-free but frontal-face only; for
  side profiles or crowded/small faces, swap in a YOLO-face or a DNN face
  detector for better recall — the anonymization logic stays the same.
- The blur/pixelate strength scales automatically with face size so distant
  and close-up faces are anonymized proportionally.
