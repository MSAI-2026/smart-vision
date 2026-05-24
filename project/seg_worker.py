"""
seg_worker.py -- Background thread that runs SegFormer-B0 on the latest frame.

The main loop submits frames with submit(); the worker always processes the most
recent submission (older queued frames are dropped). The latest result is read
non-blockingly with latest(). This decouples ~100 ms segmentation from the
~30 ms-per-frame main loop.
"""

import threading
import time

import numpy as np
from PIL import Image

from project.model import segment


class SegmentationWorker:
    """Latest-frame-wins segmentation pump on a daemon thread."""

    def __init__(self, processor, model, device) -> None:
        self._processor = processor
        self._model = model
        self._device = device

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending_rgb: np.ndarray | None = None
        self._pending_id: int = 0

        self._latest_result: dict | None = None
        self._latest_frame_rgb: np.ndarray | None = None  # frame that produced it
        self._latest_id: int = -1

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame_rgb: np.ndarray) -> None:
        """Hand a fresh RGB frame to the worker (replaces any pending one)."""
        with self._cond:
            self._pending_rgb = frame_rgb
            self._pending_id += 1
            self._cond.notify()

    def latest(self) -> tuple[dict | None, np.ndarray | None, int]:
        """Return (result, reference_frame_rgb, frame_id) without blocking."""
        with self._lock:
            return self._latest_result, self._latest_frame_rgb, self._latest_id

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._running and self._pending_rgb is None:
                    self._cond.wait()
                if not self._running:
                    return
                frame_rgb = self._pending_rgb
                frame_id = self._pending_id
                self._pending_rgb = None

            try:
                pil = Image.fromarray(frame_rgb)
                result = segment(pil, self._processor, self._model, self._device)
            except Exception as exc:  # noqa: BLE001 - keep worker alive
                # Log and continue; one bad frame shouldn't kill the pipeline.
                print(f"[seg_worker] inference failed: {exc}")
                time.sleep(0.05)
                continue

            with self._lock:
                self._latest_result = result
                self._latest_frame_rgb = frame_rgb
                self._latest_id = frame_id

    def shutdown(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify()
        self._thread.join(timeout=2.0)
