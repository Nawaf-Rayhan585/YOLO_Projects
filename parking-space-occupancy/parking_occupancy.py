"""
Parking Space Occupancy
Loads parking-slot polygons defined with define_parking_slots.py and marks
each one occupied (red) or free (green) in real time based on whether a
detected vehicle's center point falls inside it, with a live free/total
counter.

Usage:
    python parking_occupancy.py --source lot.mp4 --slots slots.json
"""

import argparse
import json

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO-based parking space occupancy detector")
    parser.add_argument("--source", default="0", help="Video path or webcam index (default: 0)")
    parser.add_argument("--slots", default="slots.json", help="JSON file of slot polygons from define_parking_slots.py")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model weights")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--no-show", action="store_true", help="Don't open a preview window")
    return parser.parse_args()


def open_source(source: str):
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


def main():
    args = parse_args()
    with open(args.slots) as f:
        raw_slots = json.load(f)
    slots = [np.array(slot, dtype=np.int32) for slot in raw_slots]
    if not slots:
        raise SystemExit(f"No slots found in {args.slots} — run define_parking_slots.py first")

    model = YOLO(args.model)
    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, classes=VEHICLE_CLASSES, conf=args.conf, verbose=False)[0]
        vehicle_points = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            vehicle_points.append(((x1 + x2) / 2, (y1 + y2) / 2))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 1)

        occupied_count = 0
        for slot in slots:
            occupied = any(cv2.pointPolygonTest(slot, pt, False) >= 0 for pt in vehicle_points)
            color = (0, 0, 255) if occupied else (0, 255, 0)
            if occupied:
                occupied_count += 1
            cv2.polylines(frame, [slot], True, color, 2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [slot], color)
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        free = len(slots) - occupied_count
        cv2.rectangle(frame, (0, 0), (260, 45), (0, 0, 0), -1)
        cv2.putText(frame, f"Free: {free} / {len(slots)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if not args.no_show:
            cv2.imshow("Parking Space Occupancy", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
