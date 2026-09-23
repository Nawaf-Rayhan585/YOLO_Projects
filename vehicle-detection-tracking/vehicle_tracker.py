"""
Vehicle Detection & Tracking
Detects and tracks cars, motorcycles, buses and trucks in a video or webcam
feed with YOLO + ByteTrack, keeping a persistent ID per vehicle and a
running count per class.

Usage:
    python vehicle_tracker.py --source cars.mp4
    python vehicle_tracker.py --source 0                 # webcam
    python vehicle_tracker.py --source cars.mp4 --save output.mp4
"""

import argparse
import time
from collections import defaultdict

import cv2
import supervision as sv
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack vehicle detection and tracking")
    parser.add_argument("--source", default="0", help="Video path or webcam index (default: 0)")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--save", default=None, help="Optional path to save annotated output video")
    parser.add_argument("--no-show", action="store_true", help="Don't open a preview window")
    return parser.parse_args()


def open_source(source: str):
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


def main():
    args = parse_args()
    model = YOLO(args.model)
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w, h = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    seen_ids_by_class = defaultdict(set)
    start = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        results = model(frame, classes=list(VEHICLE_CLASSES.keys()), conf=args.conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        labels = []
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
            class_name = VEHICLE_CLASSES.get(int(class_id), "vehicle")
            if tracker_id is not None:
                seen_ids_by_class[class_name].add(int(tracker_id))
            labels.append(f"{class_name} #{tracker_id}")

        annotated = box_annotator.annotate(frame.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels)

        fps_live = frames / (time.time() - start)
        y = 30
        cv2.putText(annotated, f"FPS: {fps_live:.1f}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for class_name, ids in seen_ids_by_class.items():
            y += 28
            cv2.putText(annotated, f"{class_name}: {len(ids)}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if writer:
            writer.write(annotated)
        if not args.no_show:
            cv2.imshow("Vehicle Detection & Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n===== SUMMARY =====")
    for class_name, ids in seen_ids_by_class.items():
        print(f"{class_name}: {len(ids)} unique vehicles")
    print(f"Frames processed: {frames}")


if __name__ == "__main__":
    main()
