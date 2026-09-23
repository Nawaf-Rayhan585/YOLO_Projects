"""
Suspicious Behavior Detection (pose heuristic)
Flags people who are crouching/bending AND moving their hands quickly at the
same time — a simple, explainable proxy for "reaching for something while
trying not to be seen" behavior often looked for in retail loss-prevention
footage. Built on YOLO pose estimation + per-person tracking, no training
required.

This is a heuristic demo, not a trained action classifier — tune the
thresholds below for your own camera angle and use case, or swap the scoring
function for a trained model down the line.

Usage:
    python suspicious_behavior.py --source store.mp4
    python suspicious_behavior.py --source 0
"""

import argparse
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

LEFT_WRIST, RIGHT_WRIST = 9, 10


def parse_args():
    parser = argparse.ArgumentParser(description="Pose-based suspicious behavior heuristic")
    parser.add_argument("--source", default="0", help="Video path or webcam index (default: 0)")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Ultralytics pose model weights")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--crouch-ratio", type=float, default=0.85, help="Bbox width/height ratio above which a person counts as bent/crouching")
    parser.add_argument("--move-thresh", type=float, default=0.06, help="Wrist movement per frame (as a fraction of bbox diagonal) counted as 'fast'")
    parser.add_argument("--sustain-frames", type=int, default=8, help="Consecutive suspicious frames required before flagging")
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

    prev_wrists = {}
    suspicious_streak = defaultdict(int)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(frame, persist=True, conf=args.conf, verbose=False)[0]

        if results.boxes.id is None or results.keypoints is None:
            if not args.no_show:
                cv2.imshow("Suspicious Behavior Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        track_ids = results.boxes.id.int().tolist()
        boxes = results.boxes.xyxy.tolist()
        keypoints = results.keypoints.xy.tolist()

        for track_id, box, kpts in zip(track_ids, boxes, keypoints):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            diag = float(np.hypot(w, h))

            is_crouching = (w / h) > args.crouch_ratio

            lw, rw = kpts[LEFT_WRIST], kpts[RIGHT_WRIST]
            wrist_now = np.array([(lw[0] + rw[0]) / 2, (lw[1] + rw[1]) / 2])
            wrist_prev = prev_wrists.get(track_id)
            prev_wrists[track_id] = wrist_now

            is_fast = False
            if wrist_prev is not None and diag > 0:
                displacement = np.linalg.norm(wrist_now - wrist_prev) / diag
                is_fast = displacement > args.move_thresh

            if is_crouching and is_fast:
                suspicious_streak[track_id] += 1
            else:
                suspicious_streak[track_id] = max(0, suspicious_streak[track_id] - 1)

            flagged = suspicious_streak[track_id] >= args.sustain_frames
            color = (0, 0, 255) if flagged else (0, 255, 0)
            label = f"ID {track_id}: SUSPICIOUS" if flagged else f"ID {track_id}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if not args.no_show:
            cv2.imshow("Suspicious Behavior Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
