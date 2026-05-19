"""
navigation.py -- Obstacle-avoidance direction from a segmentation map.

Splits the frame into left / center / right regions, counts safe (walkable)
pixels in each, and returns a movement direction.  Temporal smoothing via
majority vote over the last N frames reduces jitter.
"""

from collections import Counter, deque

import numpy as np

from project.overlay import CATEGORIES, _resolve_ids


class NavigationAdvisor:
    """Stateful advisor that converts segmentation maps into directions."""

    DIRECTIONS = ("MOVE LEFT", "MOVE FORWARD", "MOVE RIGHT", "STOP")

    def __init__(self, history_len: int = 5, safe_threshold: float = 0.05):
        self._history_len = history_len
        self._safe_threshold = safe_threshold
        self._history: deque[str] = deque(maxlen=history_len)
        self._walk_ids: set[int] | None = None
        self._id2label_id: int | None = None

    def _ensure_walk_ids(self, id2label: dict) -> None:
        """Cache walkable class IDs; rebuild only when the label dict changes."""
        key = id(id2label)
        if self._id2label_id == key:
            return
        label_lookup = {str(v).strip().lower(): int(k) for k, v in id2label.items()}
        self._walk_ids = _resolve_ids(CATEGORIES["walkable"], label_lookup)
        self._id2label_id = key

    def _compute_scores(self, class_map: np.ndarray) -> dict[str, float]:
        """Return safe-pixel ratio for each of the three vertical regions."""
        h, w = class_map.shape[:2]
        third = w // 3

        regions = {
            "left": class_map[:, :third],
            "center": class_map[:, third: 2 * third],
            "right": class_map[:, 2 * third:],
        }

        safe_mask = np.isin(class_map, list(self._walk_ids))
        scores: dict[str, float] = {}
        for name, region in regions.items():
            region_mask = safe_mask[:, :third] if name == "left" else (
                safe_mask[:, third: 2 * third] if name == "center"
                else safe_mask[:, 2 * third:]
            )
            scores[name] = float(region_mask.sum()) / region_mask.size

        return scores

    def _pick_direction(self, scores: dict[str, float]) -> str:
        """Decide raw direction from region scores."""
        if all(v < self._safe_threshold for v in scores.values()):
            return "STOP"

        best_region = max(scores, key=scores.get)  # type: ignore[arg-type]
        return {
            "left": "MOVE LEFT",
            "center": "MOVE FORWARD",
            "right": "MOVE RIGHT",
        }[best_region]

    def _smooth(self) -> str:
        """Majority vote over the history buffer."""
        if not self._history:
            return "STOP"
        counter = Counter(self._history)
        return counter.most_common(1)[0][0]

    def update(self, class_map: np.ndarray, id2label: dict) -> dict:
        """Process one frame and return direction + debug info.

        Returns
        -------
        dict with keys:
            direction     : str   -- smoothed direction
            raw_direction : str   -- unsmoothed single-frame direction
            scores        : dict  -- {"left": float, "center": float, "right": float}
        """
        self._ensure_walk_ids(id2label)
        scores = self._compute_scores(class_map)
        raw = self._pick_direction(scores)
        self._history.append(raw)
        smoothed = self._smooth()

        return {
            "direction": smoothed,
            "raw_direction": raw,
            "scores": scores,
        }
