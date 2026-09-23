"""
Crowd Heatmap
Builds a live, decaying heatmap of where people have been standing/moving in
a scene, overlaid on the video, alongside a running person count.

Usage:
    python crowd_heatmap.py --source people.mp4
    python crowd_heatmap.py --source 0
"""

import argparse

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO crowd density heatmap")
    parser.add_argument("--source", default="0", help="Video path or webcam index (default: 0)")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--decay", type=float, default=0.98, help="Per-frame heat decay factor (0-1, lower fades faster)")
    parser.add_argument("--radius", type=int, default=30, help="Heat blob radius per detection, in pixels")
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

    heatmap = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        if heatmap is None:
            heatmap = np.zeros((h, w), dtype=np.float32)
        else:
            heatmap *= args.decay

        results = model(frame, classes=[0], conf=args.conf, verbose=False)[0]
        people_count = len(results.boxes)

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.circle(heatmap, (cx, cy), args.radius, 1, -1)

        heat_blur = cv2.GaussianBlur(heatmap, (0, 0), 15)
        heat_norm = cv2.normalize(heat_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
        output = cv2.addWeighted(frame, 0.7, heat_color, 0.3, 0)

        cv2.putText(output, f"People: {people_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if not args.no_show:
            cv2.imshow("Crowd Heatmap", output)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
