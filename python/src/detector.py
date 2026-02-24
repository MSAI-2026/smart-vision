"""
Object detection for Smart Vision.
Uses MacBook built-in camera and YOLOv8-nano for real-time object detection.
Returns detected objects with class name, confidence, and bounding box center.
"""

import os

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


@dataclass
class DetectedObject:
    """Single detected object."""
    class_name: str
    confidence: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int

    @property
    def center_x(self) -> float:
        return (self.bbox_x1 + self.bbox_x2) / 2

    @property
    def center_y(self) -> float:
        return (self.bbox_y1 + self.bbox_y2) / 2


class ObjectDetector:
    """Camera + YOLOv8 object detector."""

    def __init__(self, camera_index=0, model_name="yolov8n.pt", conf_threshold=0.5):
        """
        Args:
            camera_index: Camera index (0 = MacBook FaceTime camera)
            model_name: YOLOv8 model filename (yolov8n.pt = nano, lightweight)
            conf_threshold: Minimum confidence for detections
        """
        self.camera_index = camera_index
        self.conf_threshold = conf_threshold
        self._cap: Optional[cv2.VideoCapture] = None
        self._model = None
        if HAS_YOLO:
            # Prefer models/ dir next to src/
            _models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            _model_path = os.path.join(_models_dir, model_name)
            if os.path.isfile(_model_path):
                model_to_load = _model_path
            else:
                model_to_load = model_name
            self._model = YOLO(model_to_load)

    def open_camera(self) -> bool:
        """Open camera. Returns True on success."""
        self._cap = cv2.VideoCapture(self.camera_index)
        return self._cap.isOpened()

    def release_camera(self):
        """Release camera."""
        if self._cap:
            self._cap.release()
            self._cap = None

    def read_frame(self) -> Optional[np.ndarray]:
        """Read one frame from camera."""
        if not self._cap or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Run object detection on frame.
        Returns list of DetectedObject with class, confidence, bbox.
        """
        if not self._model or frame is None:
            return []
        results = self._model(frame, verbose=False)[0]
        objects = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            cls_id = int(box.cls[0])
            class_name = results.names.get(cls_id, "unknown")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            objects.append(
                DetectedObject(
                    class_name=class_name,
                    confidence=conf,
                    bbox_x1=x1,
                    bbox_y1=y1,
                    bbox_x2=x2,
                    bbox_y2=y2,
                )
            )
        return objects

    def draw_detections(self, frame: np.ndarray, objects: List[DetectedObject]) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        out = frame.copy()
        for obj in objects:
            cv2.rectangle(
                out,
                (obj.bbox_x1, obj.bbox_y1),
                (obj.bbox_x2, obj.bbox_y2),
                (0, 255, 0),
                2,
            )
            label = f"{obj.class_name} {obj.confidence:.2f}"
            cv2.putText(
                out,
                label,
                (obj.bbox_x1, obj.bbox_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        return out


if __name__ == "__main__":
    detector = ObjectDetector()
    if not detector.open_camera():
        print("Failed to open camera")
        exit(1)
    try:
        while True:
            frame = detector.read_frame()
            if frame is None:
                break
            objects = detector.detect(frame)
            frame = detector.draw_detections(frame, objects)
            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.release_camera()
        cv2.destroyAllWindows()
