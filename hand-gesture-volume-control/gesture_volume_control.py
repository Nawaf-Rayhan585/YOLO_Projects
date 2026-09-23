"""
Hand Gesture Volume Control
Controls your system's master volume with a thumb-index pinch gesture,
tracked live via MediaPipe Hands — pinch closed for 0%, spread fingers apart
for 100%. Works on Windows (pycaw), macOS (osascript) and Linux (amixer).

Usage:
    python gesture_volume_control.py
"""

import argparse
import sys

import cv2
import mediapipe as mp
import numpy as np

THUMB_TIP, INDEX_TIP, WRIST, MIDDLE_MCP = 4, 8, 0, 9


def parse_args():
    parser = argparse.ArgumentParser(description="Pinch-gesture system volume control")
    parser.add_argument("--source", default="0", help="Webcam index (default: 0)")
    parser.add_argument("--min-dist", type=float, default=0.3, help="Pinch distance (relative to hand size) mapped to 0% volume")
    parser.add_argument("--max-dist", type=float, default=1.6, help="Pinch distance (relative to hand size) mapped to 100% volume")
    parser.add_argument("--dry-run", action="store_true", help="Show the UI but don't actually change system volume")
    return parser.parse_args()


class VolumeController:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.backend = None
        if dry_run:
            return
        try:
            if sys.platform == "win32":
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self._win_volume = cast(interface, POINTER(IAudioEndpointVolume))
                self.backend = "windows"
            elif sys.platform == "darwin":
                self.backend = "macos"
            else:
                self.backend = "linux"
        except Exception as e:
            print(f"Could not init volume backend ({e}), running in preview-only mode.")
            self.backend = None

    def set_volume(self, percent: float):
        percent = max(0.0, min(100.0, percent))
        if self.dry_run or self.backend is None:
            return
        if self.backend == "windows":
            self._win_volume.SetMasterVolumeLevelScalar(percent / 100, None)
        elif self.backend == "macos":
            import os
            os.system(f"osascript -e 'set volume output volume {int(percent)}'")
        elif self.backend == "linux":
            import os
            os.system(f"amixer -q set Master {int(percent)}%")


def main():
    args = parse_args()
    controller = VolumeController(dry_run=args.dry_run)

    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)
    cap = cv2.VideoCapture(int(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam: {args.source}")

    volume = 50.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            pts = np.array([(p.x * w, p.y * h) for p in lm])

            hand_scale = np.linalg.norm(pts[WRIST] - pts[MIDDLE_MCP]) or 1.0
            pinch_dist = np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]) / hand_scale

            norm = (pinch_dist - args.min_dist) / (args.max_dist - args.min_dist)
            volume = float(np.clip(norm, 0.0, 1.0) * 100)
            controller.set_volume(volume)

            thumb, index = pts[THUMB_TIP].astype(int), pts[INDEX_TIP].astype(int)
            cv2.line(frame, tuple(thumb), tuple(index), (0, 255, 0), 3)
            cv2.circle(frame, tuple(thumb), 8, (0, 0, 255), -1)
            cv2.circle(frame, tuple(index), 8, (0, 0, 255), -1)

        bar_x, bar_top, bar_bottom = 40, 80, 400
        bar_fill = int(bar_bottom - (volume / 100) * (bar_bottom - bar_top))
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 35, bar_bottom), (255, 255, 255), 2)
        cv2.rectangle(frame, (bar_x, bar_fill), (bar_x + 35, bar_bottom), (0, 255, 0), -1)
        cv2.putText(frame, f"{int(volume)}%", (bar_x - 10, bar_top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Hand Gesture Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
