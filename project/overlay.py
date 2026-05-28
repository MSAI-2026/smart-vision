"""
overlay.py -- Build a colour-tinted overlay from a segmentation class map and
draw detection boxes on top.

Segmentation colours:
    Yellow = walkable surfaces  (floor, carpet, rug, stairs)
    Blue   = obstacles / walls
    (everything else left untinted so the user sees the raw scene)

Detection boxes (YOLO) are drawn as blue rectangles with the class label.
"""

import cv2
import numpy as np

from project.detector import Detection

# ADE20K class names grouped into categories we care about.
# Names match / substring-match SegFormer ADE20K labels (e.g. windowpane, screen door).
CATEGORIES = {
    "walkable": ["floor", "carpet", "rug", "stairs", "stairway"],
    "wall":     ["wall", "door", "screen door", "window", "windowpane"],
}

# BGR colours for each category (OpenCV uses BGR, not RGB).
COLOURS_BGR = {
    "walkable": (0, 255, 255),  # yellow
    "obstacle": (255, 0, 0),    # blue
    # "wall":     (255, 0, 0),    # blue
}

# How strongly the tint is blended over the original frame (0-1).
ALPHA = 0.45

# Detection box drawing.
_BOX_COLOUR = (255, 0, 0)     # blue, BGR
_BOX_THICKNESS = 2
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.5
_LABEL_THICKNESS = 1


def _resolve_ids(category_names: list[str], id2label: dict) -> set[int]:
    """
    Map human-readable category names to the integer class IDs used by the model.
    Handles exact matches and substring matches (ADE labels can contain synonyms).
    """
    ids: set[int] = set()
    label_lookup = {str(v).strip().lower(): int(k) for k, v in id2label.items()}

    for name in category_names:
        target = name.lower().strip()
        if target in label_lookup:
            ids.add(label_lookup[target])
            continue
        for label, cid in label_lookup.items():
            if target in label:
                ids.add(cid)
                break
    return ids


def build_overlay(frame_bgr: np.ndarray, class_map: np.ndarray, id2label: dict) -> np.ndarray:
    """
    Blend category colours onto the camera frame based on the segmentation class map.
    Does NOT draw detection boxes; call draw_detections() afterwards.
    """
    overlay = frame_bgr.copy().astype(np.float32)

    walk_ids = _resolve_ids(CATEGORIES["walkable"], id2label)
    wall_ids = _resolve_ids(CATEGORIES["wall"], id2label)
    passable_ids = walk_ids | wall_ids

    def _blend(mask: np.ndarray, colour_key: str) -> None:
        if not np.any(mask):
            return
        colour = np.array(COLOURS_BGR[colour_key], dtype=np.float32)
        overlay[mask] = (1.0 - ALPHA) * overlay[mask] + ALPHA * colour

    _blend(np.isin(class_map, list(walk_ids)), "walkable")
    # _blend(np.isin(class_map, list(wall_ids)), "wall")

    return overlay.astype(np.uint8)


def draw_detections(frame_bgr: np.ndarray, detections: list[Detection]) -> None:
    """Draw YOLO detection boxes + labels on top of the overlay (in place)."""
    for d in detections:
        cv2.rectangle(
            frame_bgr,
            (d.x1, d.y1), (d.x2, d.y2),
            _BOX_COLOUR, _BOX_THICKNESS, cv2.LINE_AA,
        )
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, _LABEL_FONT, _LABEL_SCALE, _LABEL_THICKNESS,
        )
        y_text = max(th + 4, d.y1 - 4)
        cv2.rectangle(
            frame_bgr,
            (d.x1, y_text - th - 4),
            (d.x1 + tw + 4, y_text),
            _BOX_COLOUR, -1,
        )
        cv2.putText(
            frame_bgr, label,
            (d.x1 + 2, y_text - 2),
            _LABEL_FONT, _LABEL_SCALE,
            (0, 0, 0), _LABEL_THICKNESS, cv2.LINE_AA,
        )
