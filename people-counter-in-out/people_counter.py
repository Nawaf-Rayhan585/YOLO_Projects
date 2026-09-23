"""
People Counter (IN / OUT)
Counts people crossing a horizontal line in either direction using YOLO +
ByteTrack, and shows a live IN/OUT dashboard.

Usage:
    python people_counter.py --source people.mp4
    python people_counter.py --source 0 --line 240
"""

import argparse

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack people IN/OUT counter")
    parser.add_argument("--source", default="0", help="Video path or webcam index (default: 0)")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--line", type=int, default=None, help="Y-coordinate of the counting line (default: mid-frame)")
    parser.add_argument("--no-show", action="store_true", help="Don't open a preview window")
    return parser.parse_args()


def open_source(source: str):
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


def draw_dashboard(frame, up_count, down_count):
    h, w = frame.shape[:2]
    dashboard = np.ones((h + 140, w, 3), dtype=np.uint8) * 240
    cv2.rectangle(dashboard, (0, 0), (w, 60), (0, 120, 0), -1)
    cv2.putText(dashboard, "PEOPLE COUNTER", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    dashboard[60:60 + h, 0:w] = frame

    half = w // 2
    cv2.rectangle(dashboard, (0, 60 + h), (half, 60 + h + 80), (0, 0, 180), -1)
    cv2.putText(dashboard, f"OUT: {up_count}", (20, 60 + h + 52), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3)
    cv2.rectangle(dashboard, (half, 60 + h), (w, 60 + h + 80), (0, 120, 0), -1)
    cv2.putText(dashboard, f"IN: {down_count}", (half + 20, 60 + h + 52), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3)
    return dashboard


def main():
    args = parse_args()
    model = YOLO(args.model)
    tracker = sv.ByteTrack()

    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    line_y = args.line or int(cap.get(4) // 2) or 300
    up_count = 0
    down_count = 0
    track_history = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, classes=[0], conf=args.conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

        for bbox, track_id in zip(detections.xyxy, detections.tracker_id):
            if track_id is None:
                continue
            x1, y1, x2, y2 = bbox.astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            prev_y = track_history.get(track_id, cy)
            if prev_y < line_y <= cy:
                down_count += 1
            elif prev_y > line_y >= cy:
                up_count += 1
            track_history[track_id] = cy

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not args.no_show:
            cv2.imshow("People Counting Dashboard", draw_dashboard(frame, up_count, down_count))
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal count — IN: {down_count}  OUT: {up_count}")


if __name__ == "__main__":
    main()
