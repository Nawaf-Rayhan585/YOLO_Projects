"""
Document Scanner & OCR
Classic computer-vision document scanner: finds a document's edges in a
photo, perspective-warps it into a flat top-down "scan", cleans it up into a
black-and-white scanned look, and extracts its text with Tesseract OCR.

Usage:
    python document_scanner.py --source receipt.jpg
    python document_scanner.py --source 0          # webcam, press 'c' to capture and scan
"""

import argparse

import cv2
import numpy as np
import pytesseract


def parse_args():
    parser = argparse.ArgumentParser(description="Contour-based document scanner + OCR")
    parser.add_argument("--source", default="0", help="Image path or webcam index (default: 0)")
    parser.add_argument("--out", default="scanned.jpg", help="Output path for the scanned document image")
    parser.add_argument("--text-out", default="scanned.txt", help="Output path for the extracted text")
    parser.add_argument("--no-bw", action="store_true", help="Keep color output instead of the black-and-white scan look")
    return parser.parse_args()


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0], [max_width - 1, 0],
        [max_width - 1, max_height - 1], [0, max_height - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def find_document_contour(image):
    ratio = 500.0 / image.shape[0]
    small = cv2.resize(image, (int(image.shape[1] * ratio), 500))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 60, 160)
    edged = cv2.dilate(edged, None, iterations=1)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2) / ratio

    return None


def scan(image, want_bw=True):
    doc_contour = find_document_contour(image)
    if doc_contour is None:
        print("No 4-point document contour found — falling back to the full image.")
        h, w = image.shape[:2]
        doc_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

    warped = four_point_transform(image, doc_contour.astype("float32"))

    if want_bw:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15)

    return warped


def main():
    args = parse_args()

    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():
            raise SystemExit(f"Could not open webcam: {args.source}")
        print("Press 'c' to capture and scan the current frame, 'q' to quit.")
        image = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("Document Scanner - press 'c' to capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                image = frame.copy()
                break
            if key == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        if image is None:
            return
    else:
        image = cv2.imread(args.source)
        if image is None:
            raise SystemExit(f"Could not read image: {args.source}")

    scanned = scan(image, want_bw=not args.no_bw)
    cv2.imwrite(args.out, scanned)
    print(f"Scanned document saved to {args.out}")

    text = pytesseract.image_to_string(scanned)
    with open(args.text_out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Extracted text saved to {args.text_out}")
    print("\n----- OCR TEXT -----")
    print(text.strip() or "(no text detected)")

    cv2.imshow("Scanned Document", scanned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
