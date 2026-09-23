# Document Scanner & OCR

The classic "CamScanner" pipeline, from scratch with OpenCV: finds a
document's four edges in a photo, perspective-warps it flat into a
top-down view, cleans it up into a crisp black-and-white scan, and then
extracts its text with Tesseract OCR. Pure classical computer vision — no
neural network involved.

## How it works

1. The image is resized, blurred, and run through Canny edge detection.
2. Contours are sorted by area, and the first one that approximates to
   exactly 4 points (`cv2.approxPolyDP`) is taken as the document's outline.
   If none is found, the tool falls back to using the full image.
3. The 4 corners are ordered (top-left, top-right, bottom-right, bottom-left)
   and a perspective transform warps the document flat, correcting for the
   camera angle.
4. The warped image is adaptively thresholded into a clean black-and-white
   "scanned" look (skip with `--no-bw` to keep color).
5. [Tesseract](https://github.com/tesseract-ocr/tesseract) (via
   `pytesseract`) extracts the text from the scanned result.

## Setup

This one needs the Tesseract OCR **engine** installed separately (it's a
system binary, not just a Python package):

- **Windows:** install from the [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki), then make sure `tesseract.exe` is on your `PATH`
- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

Then:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python document_scanner.py --source receipt.jpg
python document_scanner.py --source 0     # webcam — press 'c' to capture and scan
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Image path or webcam index |
| `--out` | `scanned.jpg` | Output path for the scanned document image |
| `--text-out` | `scanned.txt` | Output path for the extracted OCR text |
| `--no-bw` | off | Keep color instead of the black-and-white scan look |

## Notes

- Works best with clear contrast between the document and the background
  (e.g. a white receipt on a dark desk) — busy or low-contrast backgrounds
  can confuse the edge detector, in which case it just OCRs the full frame.
