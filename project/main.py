"""
main.py -- Live camera segmentation using SegFormer + OpenCV.

Run:
    python -m project.main                         # defaults: camera 0, 30 fps cap
    python -m project.main --camera-index 1 --max-fps 15
"""

import argparse
import sys
import time

import cv2

from project.model import load_model, segment
from project.overlay import build_overlay


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time scene segmentation with a live camera feed.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Which camera device to open (default: 0).",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=30.0,
        help="Cap the processing frame-rate (default: 30).  Use 0 for no cap.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera, preferring the AVFoundation backend on macOS."""
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(index)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(camera_index: int = 0, max_fps: float = 30.0) -> None:
    """Load the model, open the camera, and run the segmentation loop."""

    # --- Load model (downloads on first run) ---
    print("Loading model...")
    processor, model, device = load_model()
    print(f"Model ready on {device}.")

    # --- Open camera ---
    cap = _open_camera(camera_index)
    if not cap.isOpened():
        msg = f"Cannot open camera {camera_index}."
        if sys.platform == "darwin":
            msg += (
                " Check System Settings -> Privacy & Security -> Camera"
                " and allow your terminal / Python app."
            )
        raise RuntimeError(msg)

    print("Camera started. Press 'q' in the window or Ctrl+C to quit.")

    # Frame-rate limiting: sleep between frames if needed.
    frame_interval = (1.0 / max_fps) if max_fps > 0 else 0.0

    try:
        while True:
            loop_start = time.perf_counter()

            # --- Grab a frame ---
            ok, frame_bgr = cap.read()
            if not ok:
                print("Camera read failed; stopping.")
                break

            # --- Segment the frame ---
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = segment(frame_rgb, processor, model, device)

            # --- Build the colour overlay ---
            overlay_bgr = build_overlay(frame_bgr, result["class_map"], result["id2label"])

            # --- Draw FPS counter (whole-loop, not just inference) ---
            loop_ms = (time.perf_counter() - loop_start) * 1000.0
            fps_text = f"FPS: {1000.0 / max(loop_ms, 1):.1f}  (infer {result['latency_ms']:.0f}ms)"
            cv2.putText(
                overlay_bgr, fps_text,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 255), 2, cv2.LINE_AA,
            )

            # --- Show the result ---
            cv2.imshow("Smart Vision", overlay_bgr)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            # --- Frame-rate cap ---
            elapsed = time.perf_counter() - loop_start
            if frame_interval > elapsed:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    """Entry point called by `python -m project.main`."""
    args = parse_args()
    run(camera_index=args.camera_index, max_fps=args.max_fps)


if __name__ == "__main__":
    main()
