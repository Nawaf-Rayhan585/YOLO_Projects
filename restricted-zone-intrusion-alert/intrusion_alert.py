"""
Restricted Zone Intrusion Alert
Watches a polygonal zone in the frame and raises a visible + audible alert
whenever a person is detected standing inside it (e.g. a restricted area,
a machine safety zone, a "staff only" section).

Usage:
    python intrusion_alert.py --source cctv.mp4
    python intrusion_alert.py --source 0 --zone 100,100 500,100 500,400 100,400
"""

import argparse
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_ZONE_RATIO = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.85), (0.25, 0.85)]


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO restricted-zone intrusion alert")
    parser.add_argument("--source", default="0", help="Video path or webcam index (default: 0)")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model weights")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument(
        "--zone", nargs="+", default=None,
        help="Zone polygon as 'x,y' pairs in pixels, e.g. --zone 100,100 500,100 500,400 100,400. "
             "Defaults to a rectangle covering the middle of the frame.",
    )
    parser.add_argument("--cooldown", type=float, default=3.0, help="Seconds between repeated alerts")
    parser.add_argument("--mute", action="store_true", help="Disable the audio beep")
    parser.add_argument("--no-show", action="store_true", help="Don't open a preview window")
    return parser.parse_args()


def open_source(source: str):
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


def build_zone(args_zone, width, height):
    if args_zone:
        return [tuple(map(int, p.split(","))) for p in args_zone]
    return [(int(rx * width), int(ry * height)) for rx, ry in DEFAULT_ZONE_RATIO]


def beep():
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1000, 250)
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

    width, height = int(cap.get(3)) or 640, int(cap.get(4)) or 480
    zone = build_zone(args.zone, width, height)
    zone_poly = np.array(zone, dtype=np.int32)

    last_alert_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, classes=[0], conf=args.conf, verbose=False)[0]
        boxes = results.boxes

        intrusion = False
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            foot_point = ((x1 + x2) // 2, y2)

            inside = cv2.pointPolygonTest(zone_poly, foot_point, False) >= 0
            color = (0, 0, 255) if inside else (0, 255, 0)
            if inside:
                intrusion = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, foot_point, 4, color, -1)

        zone_color = (0, 0, 255) if intrusion else (0, 200, 255)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone_poly], zone_color)
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        cv2.polylines(frame, [zone_poly], True, zone_color, 2)

        if intrusion:
            cv2.putText(frame, "INTRUSION DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            now = time.time()
            if now - last_alert_time > args.cooldown:
                last_alert_time = now
                if not args.mute:
                    beep()
                print(f"[ALERT] Person entered restricted zone at {time.strftime('%H:%M:%S')}")

        if not args.no_show:
            cv2.imshow("Restricted Zone Intrusion Alert", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
