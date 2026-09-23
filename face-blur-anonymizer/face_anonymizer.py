"""
Face Blur Anonymizer
Detects every face in a video/webcam feed and blurs or pixelates it in real
time — useful for privacy-compliant footage (publishing security clips,
GDPR-friendly recordings, blurring bystanders) without training anything.
Runs fully offline with OpenCV's built-in face detector, no model download.

Usage:
    python face_anonymizer.py --source crowd.mp4
    python face_anonymizer.py --source 0 --mode pixelate
    python face_anonymizer.py --source crowd.mp4 --save anonymized.mp4
"""

import argparse

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time face blur/pixelate anonymizer")
    parser.add_argument("--source", default="0", help="Video path or webcam index (default: 0)")
    parser.add_argument("--mode", choices=["blur", "pixelate", "box"], default="blur", help="Anonymization style")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Haar cascade scale factor")
    parser.add_argument("--min-neighbors", type=int, default=6, help="Haar cascade min neighbors (higher = fewer false positives)")
    parser.add_argument("--save", default=None, help="Optional path to save the anonymized output video")
    parser.add_argument("--no-show", action="store_true", help="Don't open a preview window")
    return parser.parse_args()


def open_source(source: str):
    return cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)


def anonymize(frame, x, y, w, h, mode):
    face = frame[y:y + h, x:x + w]
    if face.size == 0:
        return

    if mode == "blur":
        k = max(15, (min(w, h) // 2) | 1)
        frame[y:y + h, x:x + w] = cv2.GaussianBlur(face, (k, k), 0)
    elif mode == "pixelate":
        small = cv2.resize(face, (max(1, w // 12), max(1, h // 12)), interpolation=cv2.INTER_LINEAR)
        frame[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)


def main():
    args = parse_args()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w, h = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    total_faces = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=args.scale_factor, minNeighbors=args.min_neighbors)

        for (x, y, w, h) in faces:
            total_faces += 1
            anonymize(frame, x, y, w, h, args.mode)

        cv2.putText(frame, f"Faces anonymized: {len(faces)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if writer:
            writer.write(frame)
        if not args.no_show:
            cv2.imshow("Face Blur Anonymizer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\nTotal face detections processed: {total_faces}")


if __name__ == "__main__":
    main()
