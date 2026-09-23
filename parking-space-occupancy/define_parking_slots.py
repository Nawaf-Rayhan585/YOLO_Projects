"""
Parking Slot Definer
Interactive tool to mark out parking-space polygons on a reference frame by
clicking their corners, then saves them to a JSON file for
parking_occupancy.py to use.

Usage:
    python define_parking_slots.py --image lot.jpg --out slots.json
    python define_parking_slots.py --video lot.mp4 --out slots.json   # uses the first frame

Controls:
    Left click   - add a corner to the current slot
    n            - finish current slot (needs >= 3 points) and start a new one
    z            - undo the last point
    s            - save all completed slots to the output file
    q            - quit (auto-saves if there are completed slots)
"""

import argparse
import json

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Click out parking slot polygons on a reference frame")
    parser.add_argument("--image", default=None, help="Path to a reference image")
    parser.add_argument("--video", default=None, help="Path to a reference video (first frame is used)")
    parser.add_argument("--out", default="slots.json", help="Output JSON file for the slot polygons")
    return parser.parse_args()


def load_reference_frame(args):
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read image: {args.image}")
        return frame
    if args.video:
        cap = cv2.VideoCapture(args.video)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"Could not read first frame of: {args.video}")
        return frame
    raise SystemExit("Provide either --image or --video")


def main():
    args = parse_args()
    base_frame = load_reference_frame(args)

    slots = []
    current = []

    def redraw():
        frame = base_frame.copy()
        for slot in slots:
            cv2.polylines(frame, [cv2_points(slot)], True, (0, 255, 0), 2)
        if len(current) >= 2:
            cv2.polylines(frame, [cv2_points(current)], False, (0, 165, 255), 2)
        for x, y in current:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"Slots: {len(slots)}  |  n=finish slot  s=save  z=undo  q=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Define Parking Slots", frame)

    def cv2_points(pts):
        import numpy as np
        return np.array(pts, dtype=int)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
            redraw()

    cv2.namedWindow("Define Parking Slots")
    cv2.setMouseCallback("Define Parking Slots", on_click)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("n"):
            if len(current) >= 3:
                slots.append(current.copy())
                current.clear()
                redraw()
            else:
                print("Need at least 3 points to finish a slot.")

        elif key == ord("z"):
            if current:
                current.pop()
            elif slots:
                current = slots.pop()
            redraw()

        elif key == ord("s"):
            with open(args.out, "w") as f:
                json.dump(slots, f)
            print(f"Saved {len(slots)} slots to {args.out}")

        elif key == ord("q"):
            if slots:
                with open(args.out, "w") as f:
                    json.dump(slots, f)
                print(f"Saved {len(slots)} slots to {args.out}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
