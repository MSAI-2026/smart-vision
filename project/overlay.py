"""
overlay.py -- Build a colour-tinted overlay from a segmentation class map.

Colours:
    Green  = walkable surfaces  (floor, carpet, rug, stairs)
    Red    = obstacles          (any class not walkable or wall)
    Blue   = walls              (wall, door, window)
"""

import numpy as np

# ADE20K class names grouped into categories we care about.
# Names match / substring-match SegFormer ADE20K labels (e.g. windowpane, screen door).
CATEGORIES = {
    "walkable": ["floor", "carpet", "rug", "stairs", "stairway"],
    "wall":     ["wall", "door", "screen door", "window", "windowpane"],
}

# BGR colours for each category (OpenCV uses BGR, not RGB).
COLOURS_BGR = {
    "walkable": (0, 255, 0),    # green
    "obstacle": (0, 0, 255),    # red
    "wall":     (255, 0, 0),    # blue
}

# How strongly the tint is blended over the original frame (0-1).
ALPHA = 0.45


def _resolve_ids(category_names: list[str], id2label: dict) -> set[int]:
    """
    Map human-readable category names to the integer class IDs used by the model.
    Handles exact matches and substring matches (ADE labels can contain synonyms).
    """
    ids: set[int] = set()
    # Build a lowercase lookup from the model's label map.
    label_lookup = {str(v).strip().lower(): int(k) for k, v in id2label.items()}

    for name in category_names:
        target = name.lower().strip()
        # Try exact match first.
        if target in label_lookup:
            ids.add(label_lookup[target])
            continue
        # Fall back to substring match.
        for label, cid in label_lookup.items():
            if target in label:
                ids.add(cid)
                break
    return ids


def build_overlay(frame_bgr: np.ndarray, class_map: np.ndarray, id2label: dict) -> np.ndarray:
    """
    Blend category colours onto the camera frame based on the segmentation class map.

    Args:
        frame_bgr  -- original camera frame in BGR (from cv2)
        class_map  -- (H, W) array of predicted class IDs
        id2label   -- model's class-ID-to-label mapping

    Returns:
        BGR image (same size) with coloured segmentation overlay.
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

    # Red first, then blue, then green so walkable/wall win over the obstacle complement.
    _blend(~np.isin(class_map, list(passable_ids)), "obstacle")
    _blend(np.isin(class_map, list(wall_ids)), "wall")
    _blend(np.isin(class_map, list(walk_ids)), "walkable")

    return overlay.astype(np.uint8)
