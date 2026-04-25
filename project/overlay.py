"""
overlay.py -- Build a colour-tinted overlay from a segmentation class map.

Colours:
    Green  = walkable surfaces  (floor, carpet, rug)
    Red    = obstacles          (chair, table, sofa, bed, person)
    Blue   = walls
"""

import numpy as np

# ADE20K class names grouped into categories we care about.
CATEGORIES = {
    "walkable": ["floor", "carpet", "rug"],
    "obstacle": ["chair", "table", "sofa", "bed", "person"],
    "wall":     ["wall"],
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

    for category, names in CATEGORIES.items():
        # Find which pixel positions belong to this category.
        cat_ids = _resolve_ids(names, id2label)
        if not cat_ids:
            continue

        mask = np.isin(class_map, list(cat_ids))
        colour = np.array(COLOURS_BGR[category], dtype=np.float32)

        # Blend: out = (1 - alpha) * original + alpha * colour
        overlay[mask] = (1.0 - ALPHA) * overlay[mask] + ALPHA * colour

    return overlay.astype(np.uint8)
