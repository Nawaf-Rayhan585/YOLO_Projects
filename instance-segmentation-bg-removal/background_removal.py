"""
Instance Segmentation Background Removal
Uses a YOLO segmentation model to cut people out of the frame in real time
and replace the background with a blur, a solid color, or a custom image —
a green-screen effect with no green screen.

Usage:
    python background_removal.py --source 0 --mode blur
    python background_removal.py --source call.mp4 --mode image --bg office.jpg
"""

import argparse

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO segmentation background removal")
    parser.add_argument("--source", default="0", help="Video path or webcam index (default: 0)")
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Ultralytics segmentation model weights")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--class-id", type=int, default=0, help="COCO class to keep in the foreground (default: 0 = person)")
    parser.add_argument("--mode", choices=["blur", "color", "image"], default="blur", help="Background replacement style")
    parser.add_argument("--color", default="0,255,0", help="BGR color for --mode color, e.g. '0,255,0'")
    parser.add_argument("--bg", default=None, help="Background image path for --mode image")
    parser.add_argument("--no-show", action="store_true", help="Don't open a preview window")
    return parser.parse_args()


def open_source(source: str):
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


def build_background(mode, args, shape):
    h, w = shape[:2]
    if mode == "color":
        b, g, r = map(int, args.color.split(","))
        return np.full((h, w, 3), (b, g, r), dtype=np.uint8)
    if mode == "image":
        if not args.bg:
            raise SystemExit("--mode image requires --bg <path>")
        img = cv2.imread(args.bg)
        if img is None:
            raise SystemExit(f"Could not read background image: {args.bg}")
        return cv2.resize(img, (w, h))
    return None  # blur is computed per-frame


def main():
    args = parse_args()
    model = YOLO(args.model)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    static_bg = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        if args.mode != "blur" and static_bg is None:
            static_bg = build_background(args.mode, args, frame.shape)

        results = model(frame, classes=[args.class_id], conf=args.conf, verbose=False)[0]

        mask = np.zeros((h, w), dtype=np.uint8)
        if results.masks is not None:
            for seg in results.masks.xy:
                cv2.fillPoly(mask, [seg.astype(np.int32)], 255)

        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if args.mode == "blur":
            background = cv2.GaussianBlur(frame, (55, 55), 0)
        else:
            background = static_bg

        output = np.where(mask_3ch > 0, frame, background)

        if not args.no_show:
            cv2.imshow("Background Removal", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
