"""
License Plate Recognition (ANPR)
Detects license plates with a YOLO model (trained with
train_plate_detector.ipynb) and reads the plate text off each detection
with EasyOCR.

Usage:
    python anpr_inference.py --model best.pt --source parking_lot.mp4
    python anpr_inference.py --model best.pt --source 0
"""

import argparse
import csv
import time

import cv2
import easyocr
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + EasyOCR license plate recognition")
    parser.add_argument("--model", default="best.pt", help="Path to your trained plate-detection weights")
    parser.add_argument("--source", default="0", help="Video path, image path, or webcam index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--ocr-conf", type=float, default=0.4, help="Minimum OCR confidence to accept a plate reading")
    parser.add_argument("--log", default="plates.csv", help="CSV file to log recognized plates to")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR if available")
    parser.add_argument("--no-show", action="store_true", help="Don't open a preview window")
    return parser.parse_args()


def open_source(source: str):
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


def clean_plate_text(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def main():
    args = parse_args()
    model = YOLO(args.model)
    reader = easyocr.Reader(["en"], gpu=args.gpu)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    log_file = open(args.log, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["timestamp", "plate_text", "detection_conf", "ocr_conf"])

    seen_plates = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=args.conf, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_conf = float(box.conf[0])
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue

            ocr_results = reader.readtext(crop)
            plate_text, ocr_conf = "", 0.0
            for _, text, conf in ocr_results:
                if conf > ocr_conf:
                    plate_text, ocr_conf = clean_plate_text(text), conf

            color = (0, 255, 0) if plate_text else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if plate_text and ocr_conf >= args.ocr_conf:
                cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                key = (plate_text, x1 // 50, y1 // 50)
                if key not in seen_plates:
                    seen_plates.add(key)
                    csv_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), plate_text, f"{det_conf:.2f}", f"{ocr_conf:.2f}"])
                    print(f"[PLATE] {plate_text}  (det={det_conf:.2f}, ocr={ocr_conf:.2f})")

        if not args.no_show:
            cv2.imshow("License Plate Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    log_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
