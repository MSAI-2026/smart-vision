"""
overlay.py -- Build a colour-tinted overlay from a segmentation class map.

Colours:
    Green  = walkable surfaces  (floor, carpet, rug, stairs)
    Red    = obstacles          (any class not walkable or wall)
    Blue   = walls              (wall, door, window)
"""

import cv2
import numpy as np

CATEGORIES = {
    "walkable": ["floor", "carpet", "rug", "stairs", "stairway"],
    "wall":     ["wall", "door", "screen door", "window", "windowpane"],
}

# BGR colours for each category.
COLOURS_BGR = {
    "walkable": (0, 255, 0),    # green
    "obstacle": (0, 0, 255),    # red
    "wall":     (255, 0, 0),    # blue
}

ALPHA = 0.45

# Cache the per-id colour palette so we don't rebuild it every frame.
_palette_cache: dict[int, np.ndarray] = {}


def _resolve_ids(category_names: list[str], label_lookup: dict[str, int]) -> set[int]:
    """Map category name list -> set of class IDs, with substring fallback."""
    ids: set[int] = set()
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


def _build_palette(id2label: dict) -> np.ndarray:
    """Return a (num_classes, 3) BGR palette mapping each class ID to its overlay colour."""
    label_lookup = {str(v).strip().lower(): int(k) for k, v in id2label.items()}
    walk_ids = _resolve_ids(CATEGORIES["walkable"], label_lookup)
    wall_ids = _resolve_ids(CATEGORIES["wall"], label_lookup)

    num_classes = max(int(k) for k in id2label.keys()) + 1
    palette = np.empty((num_classes, 3), dtype=np.uint8)
    palette[:] = COLOURS_BGR["obstacle"]  # default everything to red
    for cid in wall_ids:
        palette[cid] = COLOURS_BGR["wall"]
    for cid in walk_ids:
        palette[cid] = COLOURS_BGR["walkable"]
    return palette


def build_overlay(frame_bgr: np.ndarray, class_map: np.ndarray, id2label: dict) -> np.ndarray:
    """Blend a category colour tint onto the camera frame."""
    key = id(id2label)
    palette = _palette_cache.get(key)
    if palette is None:
        palette = _build_palette(id2label)
        _palette_cache[key] = palette

    # Single LUT lookup + one blended write — replaces 3 np.isin + 3 float blends.
    colour_layer = palette[class_map]
    return cv2.addWeighted(frame_bgr, 1.0 - ALPHA, colour_layer, ALPHA, 0)
