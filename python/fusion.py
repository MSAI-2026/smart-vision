"""
Fusion logic for Smart Vision.
Combines IMU orientation (yaw, pitch, roll) with camera object detection.
Object whose center is closest to image center = "what the user is looking at".
LiDAR provides distance to that object.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from detector import DetectedObject


@dataclass
class FusedResult:
    """Result of fusing sensors with object detection."""
    object_class: str
    confidence: float
    distance_cm: int
    bbox_center: Tuple[float, float]


def fuse(
    objects: List[DetectedObject],
    distance_cm: int,
    frame_width: int,
    frame_height: int,
    center_region_ratio: float = 0.4,
) -> Optional[FusedResult]:
    """
    Select the object the user is looking at (closest to image center)
    and attach LiDAR distance.

    Args:
        objects: Detected objects from camera
        distance_cm: LiDAR distance in cm
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        center_region_ratio: Only consider objects whose center falls within
            this fraction of frame center (0.4 = middle 40% region).

    Returns:
        FusedResult if an object is in the center region, else None.
    """
    if not objects or frame_width <= 0 or frame_height <= 0:
        return None

    cx = frame_width / 2
    cy = frame_height / 2
    half_w = frame_width * center_region_ratio / 2
    half_h = frame_height * center_region_ratio / 2

    # Filter: only objects whose center is in the central region
    candidates = []
    for obj in objects:
        if (
            abs(obj.center_x - cx) <= half_w
            and abs(obj.center_y - cy) <= half_h
        ):
            dist = ((obj.center_x - cx) ** 2 + (obj.center_y - cy) ** 2) ** 0.5
            candidates.append((dist, obj))

    if not candidates:
        return None

    # Pick closest to center
    candidates.sort(key=lambda x: x[0])
    _, best = candidates[0]

    return FusedResult(
        object_class=best.class_name,
        confidence=best.confidence,
        distance_cm=distance_cm,
        bbox_center=(best.center_x, best.center_y),
    )
