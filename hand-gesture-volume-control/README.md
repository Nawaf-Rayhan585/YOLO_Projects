# Hand Gesture Volume Control

Controls your computer's master volume with a pinch gesture — bring your
thumb and index finger together for 0%, spread them apart for 100% — tracked
live via [MediaPipe Hands](https://google.github.io/mediapipe/). A small,
fun human-computer-interaction demo with zero training required.

## How it works

1. MediaPipe Hands tracks 21 landmarks on one hand per frame.
2. The distance between the thumb tip and index fingertip is measured, then
   normalized by the wrist-to-middle-knuckle distance so it works the same
   whether your hand is close to or far from the camera.
3. That normalized pinch distance is linearly mapped from `--min-dist`
   (closed pinch → 0%) to `--max-dist` (open hand → 100%) and applied to the
   system's master volume live.

Volume is set through:
- **Windows** — [`pycaw`](https://github.com/AndreMiras/pycaw) (Core Audio API)
- **macOS** — `osascript -e "set volume output volume N"`
- **Linux** — `amixer -q set Master N%`

## Usage

```bash
pip install -r requirements.txt

python gesture_volume_control.py
python gesture_volume_control.py --dry-run   # preview only, doesn't touch system volume
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index |
| `--min-dist` | `0.3` | Normalized pinch distance mapped to 0% volume |
| `--max-dist` | `1.6` | Normalized pinch distance mapped to 100% volume |
| `--dry-run` | off | Show the UI/bar without changing system volume |

Press `q` to quit.

## Notes

- On Linux, `amixer` needs to be installed and a `Master` mixer control
  needs to exist (`alsa-utils` — adjust the control name for your setup if
  needed).
- If `--min-dist`/`--max-dist` feel off for your hand size or camera
  distance, run with `--dry-run` first and watch the printed/on-screen
  percentage to re-tune them.
