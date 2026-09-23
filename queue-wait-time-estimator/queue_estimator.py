"""
Queue Length & Wait-Time Estimator
Watches a queue zone (checkout line, bank counter, ticket booth) and reports
the live number of people waiting plus a rolling average wait time, measured
from each tracked person's actual entry and exit times — not a theoretical
estimate.

Usage:
    python queue_estimator.py --source checkout.mp4
    python queue_estimator.py --source 0 --zone 100,100 500,100 500,400 100,400
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_ZONE_RATIO = [(0.2, 0.15), (0.8, 0.15), (0.8, 0.9), (0.2, 0.9)]


def parse_args():
    parser = argparse.ArgumentParser(description="Queue length and wait-time estimator")
    parser.add_argument("--source", default="0", help="Video path or webcam index (default: 0)")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--zone", nargs="+", default=None, help="Queue zone polygon as 'x,y' pairs")
    parser.add_argument("--history", type=int, default=50, help="How many completed waits to average over")
    parser.add_argument("--no-show", action="store_true", help="Don't open a preview window")
    return parser.parse_args()


def open_source(source: str):
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


def build_zone(args_zone, width, height):
    if args_zone:
        return [tuple(map(int, p.split(","))) for p in args_zone]
    return [(int(rx * width), int(ry * height)) for rx, ry in DEFAULT_ZONE_RATIO]


def main():
    args = parse_args()
    model = YOLO(args.model)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    width, height = int(cap.get(3)) or 640, int(cap.get(4)) or 480
    zone_poly = np.array(build_zone(args.zone, width, height), dtype=np.int32)

    entry_time = {}
    completed_waits = deque(maxlen=args.history)
    prev_inside = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        results = model.track(frame, classes=[0], conf=args.conf, persist=True, verbose=False)[0]

        current_inside = set()
        if results.boxes.id is not None:
            track_ids = results.boxes.id.int().tolist()
            boxes = results.boxes.xyxy.tolist()

            for track_id, box in zip(track_ids, boxes):
                x1, y1, x2, y2 = box
                foot_point = (int((x1 + x2) / 2), int(y2))
                inside = cv2.pointPolygonTest(zone_poly, foot_point, False) >= 0

                color = (0, 200, 255) if inside else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                if inside:
                    current_inside.add(track_id)
                    entry_time.setdefault(track_id, now)
                    waited = now - entry_time[track_id]
                    cv2.putText(frame, f"ID {track_id}: {waited:.0f}s", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for departed_id in prev_inside - current_inside:
            if departed_id in entry_time:
                completed_waits.append(now - entry_time.pop(departed_id))
        prev_inside = current_inside

        cv2.polylines(frame, [zone_poly], True, (0, 165, 255), 2)

        avg_wait = sum(completed_waits) / len(completed_waits) if completed_waits else 0.0
        cv2.rectangle(frame, (0, 0), (300, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"Queue length: {len(current_inside)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Avg wait: {avg_wait:.1f}s ({len(completed_waits)} served)", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if not args.no_show:
            cv2.imshow("Queue Wait-Time Estimator", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
