"""
Fall Detection (elderly / workplace monitoring)
Flags a person as fallen when their body center drops sharply and their
posture goes from upright to horizontal — a pose-based heuristic that needs
no training data, built on YOLO pose estimation + per-person tracking.

Usage:
    python fall_detector.py --source room.mp4
    python fall_detector.py --source 0
"""

import argparse
import sys
import time
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12


def parse_args():
    parser = argparse.ArgumentParser(description="Pose-based fall detection heuristic")
    parser.add_argument("--source", default="0", help="Video path or webcam index (default: 0)")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Ultralytics pose model weights")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--window", type=int, default=8, help="Frames to look back over when measuring the vertical drop")
    parser.add_argument("--drop-thresh", type=float, default=0.5, help="Vertical drop (fraction of bbox height) over the window that counts as a fall trigger")
    parser.add_argument("--lying-ratio", type=float, default=1.2, help="Bbox width/height above which posture counts as horizontal/lying")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Seconds between repeated alerts for the same person")
    parser.add_argument("--mute", action="store_true", help="Disable the audio alarm")
    parser.add_argument("--no-show", action="store_true", help="Don't open a preview window")
    return parser.parse_args()


def open_source(source: str):
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


def beep():
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1400, 400)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


def main():
    args = parse_args()
    model = YOLO(args.model)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    history = defaultdict(lambda: deque(maxlen=args.window))
    fallen = {}
    last_alert = defaultdict(float)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(frame, persist=True, conf=args.conf, verbose=False)[0]

        if results.boxes.id is not None and results.keypoints is not None:
            track_ids = results.boxes.id.int().tolist()
            boxes = results.boxes.xyxy.tolist()
            keypoints = results.keypoints.xy.tolist()

            for track_id, box, kpts in zip(track_ids, boxes, keypoints):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue

                shoulder_y = (kpts[LEFT_SHOULDER][1] + kpts[RIGHT_SHOULDER][1]) / 2
                hip_y = (kpts[LEFT_HIP][1] + kpts[RIGHT_HIP][1]) / 2
                center_y = (shoulder_y + hip_y) / 2 if (shoulder_y and hip_y) else (y1 + y2) / 2

                hist = history[track_id]
                hist.append(center_y)

                aspect_ratio = w / h
                lying = aspect_ratio > args.lying_ratio

                dropped = False
                if len(hist) == hist.maxlen:
                    drop = (hist[-1] - hist[0]) / h
                    dropped = drop > args.drop_thresh

                if dropped and lying:
                    fallen[track_id] = True
                elif not lying:
                    fallen[track_id] = False

                is_fallen = fallen.get(track_id, False)
                color = (0, 0, 255) if is_fallen else (0, 255, 0)
                label = f"ID {track_id}: FALL DETECTED" if is_fallen else f"ID {track_id}"

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if is_fallen:
                    now = time.time()
                    if now - last_alert[track_id] > args.cooldown:
                        last_alert[track_id] = now
                        print(f"[ALERT] Person ID {track_id} fell at {time.strftime('%H:%M:%S')}")
                        if not args.mute:
                            beep()

        if any(fallen.values()):
            cv2.putText(frame, "FALL DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if not args.no_show:
            cv2.imshow("Fall Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
