"""
detector.py -- YOLOv8n wrapper for dynamic obstacle detection.

Runs every frame and returns bounding boxes for COCO classes that matter for
indoor navigation (person, chair, couch, etc). The navigation logic uses these
boxes to mark "occupied floor" even where the segmentation says the ground is
walkable.
"""

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from project.mps_lock import MPS_LOCK


class _NullCtx:
    """No-op context manager for non-MPS devices."""
    def __enter__(self): return None
    def __exit__(self, *exc): return False

# Default checkpoint - downloaded into the ultralytics cache on first use.
MODEL_NAME = "yolov8n.pt"

# COCO classes relevant for indoor obstacle avoidance. The rest are filtered out
# so the navigation layer doesn't react to e.g. "bird" or "stop sign".
INDOOR_OBSTACLE_CLASSES = frozenset({
    "person", "bicycle", "backpack", "handbag", "suitcase",
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
})


@dataclass
class Detection:
    """One YOLO detection."""
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


def _pick_device() -> str:
    """ultralytics expects a string device spec."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ObjectDetector:
    """YOLOv8n wrapper - frames in, list of Detection out."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        conf_threshold: float = 0.35,
        input_size: int = 320,
        device: str | None = None,
    ):
        self._device = device or _pick_device()
        self._conf = conf_threshold
        self._imgsz = input_size
        # Half precision on accelerator devices is roughly free quality-wise
        # and meaningfully faster.
        self._half = self._device in ("cuda", "mps")
        self._model = YOLO(model_name)
        # Warm-up: first call compiles kernels; do it once here.
        dummy = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        with self._maybe_lock():
            self._model.predict(
                dummy, imgsz=input_size, device=self._device,
                half=self._half, verbose=False,
            )

    def _maybe_lock(self):
        """Serialise MPS submissions with the segmentation worker."""
        return MPS_LOCK if self._device == "mps" else _NullCtx()

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run YOLO on a BGR frame. Returns detections filtered to indoor classes."""
        with self._maybe_lock():
            results = self._model.predict(
                frame_bgr,
                imgsz=self._imgsz,
                device=self._device,
                half=self._half,
                conf=self._conf,
                verbose=False,
            )[0]

        names = results.names
        out: list[Detection] = []
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy().astype(np.int32)
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, cls_ids):
            class_name = names.get(cid, str(cid))
            if class_name not in INDOOR_OBSTACLE_CLASSES:
                continue
            out.append(Detection(
                class_name=class_name,
                confidence=float(conf),
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
            ))
        return out


def boxes_to_mask(detections: list[Detection], h: int, w: int) -> np.ndarray:
    """Rasterise detection boxes into a (H, W) boolean obstacle mask."""
    mask = np.zeros((h, w), dtype=bool)
    for d in detections:
        x1 = max(0, min(w, d.x1))
        x2 = max(0, min(w, d.x2))
        y1 = max(0, min(h, d.y1))
        y2 = max(0, min(h, d.y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask
