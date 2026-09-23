"""
Fire Detection — Inference
Runs a fire-detection YOLO model (trained with train_fire_detector.ipynb)
against a video, webcam, or image, and raises an on-screen + audible alarm
whenever fire is spotted.

Usage:
    python fire_detect.py --model best.pt --source warehouse.mp4
    python fire_detect.py --model best.pt --source 0
"""

import argparse
import sys
import time

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO fire/smoke detection inference")
    parser.add_argument("--model", default="best.pt", help="Path to your trained fire-detection weights")
    parser.add_argument("--source", default="0", help="Video path, image path, or webcam index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Seconds between repeated alarms")
    parser.add_argument("--mute", action="store_true", help="Disable the audio alarm")
    parser.add_argument("--no-show", action="store_true", help="Don't open a preview window")
    return parser.parse_args()


def open_source(source: str):
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


def beep():
    try:
        if sys.platform == "win32":
            import winsound
            for _ in range(3):
                winsound.Beep(1500, 200)
        else:
            for _ in range(3):
                print("\a", end="", flush=True)
    except Exception:
        pass


def main():
    args = parse_args()
    model = YOLO(args.model)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    last_alarm = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=args.conf, verbose=False)[0]
        annotated = results.plot()

        fire_detected = len(results.boxes) > 0
        if fire_detected:
            cv2.putText(annotated, "FIRE DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            now = time.time()
            if now - last_alarm > args.cooldown:
                last_alarm = now
                if not args.mute:
                    beep()
                print(f"[ALERT] Fire detected at {time.strftime('%H:%M:%S')}")

        if not args.no_show:
            cv2.imshow("Fire Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
