"""
main.py -- Live indoor navigation pipeline.

Two-branch design:
  * SegFormer-B0 (ADE20K) runs on a background worker thread for floor/window
    segmentation. Latest mask is reused between runs, warped onto the current
    frame via optical flow so it tracks camera motion.
  * YOLOv8n runs synchronously every frame for dynamic obstacle detection
    (people, chairs, etc).
  * The fusion lives in NavigationAdvisor: walkable = floor AND not in any box.

Run:
    python -m project.main                         # camera 0, 30 fps cap
    python -m project.main --camera-index 1 --max-fps 15
"""

import argparse
import sys
import time
from collections import deque

import cv2

from project.audio import AudioNavigator
from project.detector import ObjectDetector
from project.display import draw_navigation_hints
from project.flow import warp_class_map
from project.model import load_model
from project.navigation import NavigationAdvisor
from project.overlay import build_overlay, draw_detections
from project.seg_worker import SegmentationWorker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time indoor navigation: SegFormer + YOLOv8.",
    )
    parser.add_argument(
        "--camera-index", type=int, default=0,
        help="Which camera device to open (default: 0).",
    )
    parser.add_argument(
        "--max-fps", type=float, default=30.0,
        help="Cap the processing frame-rate (default: 30). Use 0 for no cap.",
    )
    parser.add_argument(
        "--no-flow", action="store_true",
        help="Disable optical-flow mask warping (use stale mask as-is).",
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Camera frame width in pixels (default: 640).",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Camera frame height in pixels (default: 480).",
    )
    return parser.parse_args()


def _open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open a camera at the requested resolution, preferring AVFoundation on macOS."""
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index)

    if cap.isOpened():
        # Requests are advisory; the driver picks the closest supported mode.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera: requested {width}x{height}, got {actual_w}x{actual_h}.")
    return cap


def _draw_stats(
    frame_bgr,
    fps_actual: float,
    seg_latency_ms: float,
    det_latency_ms: float,
) -> None:
    text = (
        f"FPS: {fps_actual:5.1f}  "
        f"seg: {seg_latency_ms:5.1f} ms  "
        f"det: {det_latency_ms:5.1f} ms"
    )
    cv2.putText(
        frame_bgr, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
    )


def run(
    camera_index: int = 0,
    max_fps: float = 30.0,
    use_flow: bool = True,
    width: int = 640,
    height: int = 480,
) -> None:
    """Load models, open the camera, run the navigation loop."""

    print("Loading SegFormer-B0 (segmentation)...")
    processor, seg_model, seg_device = load_model()
    print(f"  ready on {seg_device}.")

    print("Loading YOLOv8n (detection)...")
    detector = ObjectDetector()
    print("  ready.")

    advisor = NavigationAdvisor()
    audio = AudioNavigator(cooldown_s=2.5, stable_frames=3, stop_stable_frames=6)
    seg_worker = SegmentationWorker(processor, seg_model, seg_device)

    cap = _open_camera(camera_index, width, height)
    if not cap.isOpened():
        msg = f"Cannot open camera {camera_index}."
        if sys.platform == "darwin":
            msg += (
                " Check System Settings -> Privacy & Security -> Camera"
                " and allow your terminal / Python app."
            )
        raise RuntimeError(msg)

    print("Camera started. Press 'q' in the window or Ctrl+C to quit.")
    print("Waiting for first segmentation mask...")

    frame_interval = (1.0 / max_fps) if max_fps > 0 else 0.0

    last_seg_id = -1
    cached_class_map = None
    cached_frame_rgb = None
    cached_id2label = None

    fps_window: deque[float] = deque(maxlen=30)

    try:
        while True:
            loop_start = time.perf_counter()

            ok, frame_bgr = cap.read()
            if not ok:
                print("Camera read failed; stopping.")
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Hand the latest frame to the segmentation worker (non-blocking).
            seg_worker.submit(frame_rgb)

            # --- Pull the most recent segmentation result, if any ---
            result, ref_frame_rgb, seg_id = seg_worker.latest()
            if result is not None and seg_id != last_seg_id:
                cached_class_map = result["class_map"]
                cached_frame_rgb = ref_frame_rgb
                cached_id2label = result["id2label"]
                last_seg_id = seg_id

            # --- Detection runs every frame ---
            det_start = time.perf_counter()
            detections = detector.detect(frame_bgr)
            det_latency_ms = (time.perf_counter() - det_start) * 1000.0

            # If we don't have a seg mask yet, show the raw frame + boxes.
            if cached_class_map is None or cached_id2label is None:
                display = frame_bgr.copy()
                draw_detections(display, detections)
                cv2.putText(
                    display, "loading segmentation...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.imshow("Smart Vision", display)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            # --- Warp the (potentially stale) class map onto the current frame ---
            if use_flow and cached_frame_rgb is not None:
                class_map = warp_class_map(cached_class_map, cached_frame_rgb, frame_rgb)
            else:
                class_map = cached_class_map

            # --- Build overlay + run navigation ---
            overlay_bgr = build_overlay(frame_bgr, class_map, cached_id2label)
            draw_detections(overlay_bgr, detections)

            nav = advisor.update(class_map, cached_id2label, detections)
            audio.announce(nav["direction"], nav["phrase"])
            draw_navigation_hints(overlay_bgr, nav)

            # --- FPS counter (real end-to-end, not just inference time) ---
            elapsed = time.perf_counter() - loop_start
            fps_window.append(elapsed)
            avg_dt = sum(fps_window) / len(fps_window)
            fps_actual = 1.0 / avg_dt if avg_dt > 0 else 0.0
            seg_latency_ms = (
                result["latency_ms"] if result is not None else 0.0
            )
            _draw_stats(overlay_bgr, fps_actual, seg_latency_ms, det_latency_ms)

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
        audio.shutdown()
        seg_worker.shutdown()
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    run(
        camera_index=args.camera_index,
        max_fps=args.max_fps,
        use_flow=not args.no_flow,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
