# Queue Length & Wait-Time Estimator

Watches a queue zone — a checkout line, bank counter, ticket booth — and
reports the live number of people waiting plus a rolling **average wait
time measured from real entries and exits**, not a theoretical Little's Law
guess.

## How it works

1. YOLO + ByteTrack (via `model.track(persist=True)`) detects and tracks
   people, checking each one's foot point against a queue zone polygon with
   `cv2.pointPolygonTest`.
2. The first frame a tracked person's foot point lands inside the zone, their
   entry time is recorded.
3. When a previously-inside ID is no longer inside the zone (served and
   walked away, or left the frame), their wait duration
   (`now - entry_time`) is added to a rolling history.
4. The on-screen dashboard shows the current queue length and the average of
   the last `--history` completed waits.

## Usage

```bash
pip install -r requirements.txt

python queue_estimator.py --source checkout.mp4
python queue_estimator.py --source 0 --zone 100,100 500,100 500,400 100,400
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Video file path or webcam index |
| `--model` | `yolo11n.pt` | Any Ultralytics YOLO checkpoint |
| `--conf` | `0.4` | Minimum detection confidence |
| `--zone` | mid-frame rectangle | Queue polygon as `x,y x,y ...` |
| `--history` | `50` | Number of completed waits averaged for the display |
| `--no-show` | off | Run headless, no preview window |

Press `q` to quit.

## Notes

- A person briefly occluded (blocked by someone else) can be mis-tracked as
  having "left" the queue, which will slightly undercount their wait — good
  enough for a live estimate, not for billing-grade accuracy.
- Point the zone at just the queue line itself, not the service counter, or
  service time will get counted as wait time.
