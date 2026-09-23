"""
Rat / Rodent Detection — Inference
Runs a rodent-detection YOLO model (trained with train_rat_detector.ipynb)
against a video, webcam, or image, and logs every sighting with a timestamp.

Usage:
    python rat_detect.py --model best.pt --source warehouse.mp4
    python rat_detect.py --model best.pt --source 0
"""

import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO rat/rodent detection inference")
    parser.add_argument("--model", default="best.pt", help="Path to your trained rodent-detection weights")
    parser.add_argument("--source", default="0", help="Video path, image path, or webcam index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--log", default="sightings.csv", help="CSV file to log sightings to")
    parser.add_argument("--no-show", action="store_true", help="Don't open a preview window")
    return parser.parse_args()


def open_source(source: str):
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


def main():
    args = parse_args()
    model = YOLO(args.model)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    log_file = open(args.log, "w", newline="")
    log_file.write("timestamp,confidence\n")
    total_sightings = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=args.conf, verbose=False)[0]
        annotated = results.plot()

        if len(results.boxes) > 0:
            total_sightings += len(results.boxes)
            for box in results.boxes:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{float(box.conf[0]):.2f}\n")
            cv2.putText(annotated, "RODENT DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if not args.no_show:
            cv2.imshow("Rat / Rodent Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    log_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal sightings logged: {total_sightings} -> {args.log}")


if __name__ == "__main__":
    main()
