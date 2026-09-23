"""
PPE Safety Compliance Detection
Runs a PPE-detection YOLO model (trained with train_ppe_detector.ipynb)
over a video/webcam feed and flags anyone missing required safety gear
(hard hat, safety vest, mask, etc.) with a red box and a logged violation.

Works with any Roboflow-style PPE dataset that labels violations with a
"NO-" / "no_" / "without" prefix in the class name (e.g. "NO-Hardhat",
"Safety Vest") — no hardcoded class list needed.

Usage:
    python ppe_compliance.py --model best.pt --source site.mp4
    python ppe_compliance.py --model best.pt --source 0
"""

import argparse
import csv
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO

VIOLATION_KEYWORDS = ("no-", "no_", "without", "missing")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO PPE compliance detection")
    parser.add_argument("--model", default="best.pt", help="Path to your trained PPE-detection weights")
    parser.add_argument("--source", default="0", help="Video path, image path, or webcam index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Seconds between repeated alert prints/logs")
    parser.add_argument("--log", default="violations.csv", help="CSV file to log violations to")
    parser.add_argument("--no-show", action="store_true", help="Don't open a preview window")
    return parser.parse_args()


def open_source(source: str):
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


def is_violation(class_name: str) -> bool:
    lowered = class_name.lower()
    return any(kw in lowered for kw in VIOLATION_KEYWORDS)


def main():
    args = parse_args()
    model = YOLO(args.model)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    log_file = open(args.log, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["timestamp", "violation_class", "confidence"])

    class_counter = defaultdict(int)
    last_alert = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=args.conf, verbose=False)[0]
        frame_violations = []

        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            violation = is_violation(cls_name)

            color = (0, 0, 255) if violation else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            class_counter[cls_name] += 1
            if violation:
                frame_violations.append((cls_name, conf))

        if frame_violations:
            cv2.putText(frame, f"{len(frame_violations)} PPE VIOLATION(S)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            now = time.time()
            if now - last_alert > args.cooldown:
                last_alert = now
                for cls_name, conf in frame_violations:
                    csv_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), cls_name, f"{conf:.2f}"])
                print(f"[VIOLATION] {', '.join(c for c, _ in frame_violations)} at {time.strftime('%H:%M:%S')}")

        if not args.no_show:
            cv2.imshow("PPE Safety Compliance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    log_file.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\n===== SUMMARY =====")
    for cls_name, count in sorted(class_counter.items()):
        print(f"{cls_name}: {count}")


if __name__ == "__main__":
    main()
