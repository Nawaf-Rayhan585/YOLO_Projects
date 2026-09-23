"""
Hand Type Detector (Left / Right)
Detects hands in a video or webcam feed and labels each one Left or Right,
mirror-corrected so it matches what the person themselves would call it.

Usage:
    python hand_type_detector.py --source 0
    python hand_type_detector.py --source hands.mp4
"""

import argparse

import cv2
from cvzone.HandTrackingModule import HandDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Left/Right hand detector")
    parser.add_argument("--source", default="0", help="Video path or webcam index (default: 0)")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum number of hands to track")
    parser.add_argument("--conf", type=float, default=0.8, help="Detection confidence threshold")
    parser.add_argument("--no-show", action="store_true", help="Don't open a preview window")
    return parser.parse_args()


def open_source(source: str):
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


def main():
    args = parse_args()
    detector = HandDetector(detectionCon=args.conf, maxHands=args.max_hands)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        hands, processed = detector.findHands(frame)

        for hand in hands:
            hand_type = hand.get("type", "Unknown")
            cx, cy = hand["center"]

            if hand_type not in ("Left", "Right"):
                hand_type = "Right" if cx < w // 2 else "Left"

            color = (0, 220, 0) if hand_type == "Right" else (0, 180, 255)
            font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 1.4, 3

            (tw, _), _ = cv2.getTextSize(hand_type, font, scale, thick)
            tx, ty = cx - tw // 2, cy - 40

            cv2.putText(processed, hand_type, (tx + 2, ty + 2), font, scale, (0, 0, 0), thick + 2)
            cv2.putText(processed, hand_type, (tx, ty), font, scale, color, thick)

        if not args.no_show:
            cv2.imshow("Hand Type Detector", processed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
