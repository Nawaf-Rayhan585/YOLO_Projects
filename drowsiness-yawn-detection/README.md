# Drowsiness & Yawn Detection

A driver/operator-monitoring style tool: watches a face and raises an alert
when the eyes stay closed for too long (drowsiness) or a yawn is detected —
the same core technique (Eye Aspect Ratio + Mouth Aspect Ratio) used in real
automotive driver-monitoring systems. Runs entirely offline on
[MediaPipe](https://google.github.io/mediapipe/) FaceMesh — no training, no
model download.

## How it works

1. MediaPipe FaceMesh locates 468 3D facial landmarks per frame.
2. **Eye Aspect Ratio (EAR):** for each eye, the ratio of vertical eyelid
   distance to horizontal eye-corner distance. This ratio drops sharply when
   the eye closes. If it stays below `--ear-thresh` for `--ear-frames`
   consecutive frames, a drowsiness alert fires (with a beep).
3. **Mouth Aspect Ratio (MAR):** the ratio of vertical mouth opening to
   mouth width. A sustained spike above `--mar-thresh` is flagged as a yawn.

## Usage

```bash
pip install -r requirements.txt

python drowsiness_detector.py --source 0
python drowsiness_detector.py --source dashcam.mp4
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Video file path or webcam index |
| `--ear-thresh` | `0.21` | EAR below this counts as "eyes closed" |
| `--ear-frames` | `20` | Consecutive closed-eye frames before alerting (~0.7s at 30fps) |
| `--mar-thresh` | `0.6` | MAR above this counts as a yawn |
| `--mute` | off | Disable the audio alarm |
| `--no-show` | off | Run headless, no preview window |

Press `q` to quit.

## Notes

- Works best with a front-facing, reasonably well-lit camera — the classic
  dashboard-mounted driver-facing camera angle.
- `--ear-thresh` and `--ear-frames` are the two knobs worth tuning per
  person/camera — faces and cameras vary enough that the defaults won't be
  perfect for everyone.
