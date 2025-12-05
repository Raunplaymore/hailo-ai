"""
Interactive down-the-line golf swing event labeler.

Usage:
  python tools/label_dtl_events.py --video data/dtl_raw/dtl_001.mp4 --output data/dtl_labels/dtl_001.json

Events:
  0: address (key '1')
  1: top     (key '2')
  2: impact  (key '3')
  3: finish  (key '4')

Navigation:
  d: +1 frame
  a: -1 frame
  w: +10 frames
  s: -10 frames
  enter/space: save if all events set
  q: quit without saving
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import cv2


EVENT_KEYS = {
    ord("1"): "address",
    ord("2"): "top",
    ord("3"): "impact",
    ord("4"): "finish",
}


def read_frame(cap: cv2.VideoCapture, idx: int) -> Optional:
    """Seek and read a specific frame index."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def overlay_info(frame, frame_idx: int, total: int, labels: Dict[str, Optional[int]]) -> None:
    """Draw overlay text with frame info and labeled events."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Frame {frame_idx+1}/{total}", (10, 30), font, 0.8, (0, 255, 0), 2)

    y = 60
    for name in ["address", "top", "impact", "finish"]:
        val = labels.get(name)
        txt = f"{name}: {val if val is not None else '-'}"
        cv2.putText(frame, txt, (10, y), font, 0.6, (255, 255, 255), 1)
        y += 25

    cv2.putText(frame, "a/d: -/+1   s/w: -/+10   1-4: set events   space/enter: save   q: quit",
                (10, frame.shape[0] - 20), font, 0.5, (200, 200, 0), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label 4 golf swing events on a DTL video.")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_idx = 0

    labels: Dict[str, Optional[int]] = {name: None for name in ["address", "top", "impact", "finish"]}

    while True:
        frame = read_frame(cap, current_idx)
        if frame is None:
            print(f"Failed to read frame {current_idx}.")
            break

        overlay_info(frame, current_idx, total_frames, labels)
        cv2.imshow("DTL Labeler", frame)
        key = cv2.waitKey(0)

        if key == ord("q"):
            print("Quit without saving.")
            break
        elif key in (13, 32):  # Enter or Space
            if all(v is not None for v in labels.values()):
                output = {
                    "video": video_path.name,
                    "fps": fps,
                    "num_frames": total_frames,
                    "events": labels,
                }
                out_path = Path(args.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
                print(f"Saved labels to {out_path}")
                break
            else:
                print("All events are not set yet. Label address/top/impact/finish (1-4).")
        elif key in EVENT_KEYS:
            evt = EVENT_KEYS[key]
            labels[evt] = current_idx
            print(f"Set {evt} -> frame {current_idx}")
        elif key == ord("d"):
            current_idx = min(current_idx + 1, total_frames - 1)
        elif key == ord("a"):
            current_idx = max(current_idx - 1, 0)
        elif key == ord("w"):
            current_idx = min(current_idx + 10, total_frames - 1)
        elif key == ord("s"):
            current_idx = max(current_idx - 10, 0)
        else:
            # Unmapped key, ignore and continue
            pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
