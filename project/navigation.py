"""
navigation.py -- Obstacle-avoidance direction from a segmentation map.

Scores focus on the lower frame (path at your feet), penalize obstacles in
the center column, and apply hysteresis so forward guidance does not flicker.
"""

from collections import Counter, deque

import numpy as np

from project.guidance import DIRECTION_TO_REGION, phrase_for
from project.overlay import CATEGORIES, _resolve_ids


class NavigationAdvisor:
    """Stateful advisor that converts segmentation maps into directions."""

    DIRECTIONS = ("MOVE LEFT", "MOVE FORWARD", "MOVE RIGHT", "STOP")

    def __init__(
        self,
        history_len: int = 8,
        safe_threshold: float = 0.08,
        side_margin: float = 0.15,
        hysteresis_extra: float = 0.08,
        center_obstacle_threshold: float = 0.22,
        path_top_ratio: float = 0.45,
    ):
        self._history_len = history_len
        self._safe_threshold = safe_threshold
        self._side_margin = side_margin
        self._hysteresis_extra = hysteresis_extra
        self._center_obstacle_threshold = center_obstacle_threshold
        self._path_top_ratio = path_top_ratio

        self._history: deque[str] = deque(maxlen=history_len)
        self._last_smoothed: str | None = None
        self._walk_ids: set[int] | None = None
        self._wall_ids: set[int] | None = None
        self._passable_ids: set[int] | None = None
        self._id2label_id: int | None = None

    def _ensure_class_ids(self, id2label: dict) -> None:
        """Cache class IDs; rebuild only when the label dict changes."""
        key = id(id2label)
        if self._id2label_id == key:
            return
        self._walk_ids = _resolve_ids(CATEGORIES["walkable"], id2label)
        self._wall_ids = _resolve_ids(CATEGORIES["wall"], id2label)
        self._passable_ids = self._walk_ids | self._wall_ids
        self._id2label_id = key

    def _path_slice(self, h: int) -> slice:
        """Rows from path_top to bottom -- where the user is about to step."""
        return slice(int(h * self._path_top_ratio), h)

    def _region_slices(self, h: int, w: int) -> dict[str, tuple[slice, slice]]:
        third = w // 3
        rows = self._path_slice(h)
        return {
            "left": (rows, slice(0, third)),
            "center": (rows, slice(third, 2 * third)),
            "right": (rows, slice(2 * third, w)),
        }

    def _compute_scores(self, class_map: np.ndarray) -> dict[str, float]:
        """Walkable-pixel ratio per column in the lower path band."""
        h, w = class_map.shape[:2]
        safe_mask = np.isin(class_map, list(self._walk_ids))
        scores: dict[str, float] = {}

        for name, (row_sl, col_sl) in self._region_slices(h, w).items():
            region_mask = safe_mask[row_sl, col_sl]
            scores[name] = float(region_mask.sum()) / max(region_mask.size, 1)

        return scores

    def _center_obstacle_ratio(self, class_map: np.ndarray) -> float:
        """Share of obstacle pixels in the lower center column."""
        h, w = class_map.shape[:2]
        row_sl, col_sl = self._region_slices(h, w)["center"]
        obstacle_mask = ~np.isin(class_map, list(self._passable_ids))
        region = obstacle_mask[row_sl, col_sl]
        return float(region.sum()) / max(region.size, 1)

    def _pick_direction(
        self, scores: dict[str, float], center_obstacle: float,
    ) -> tuple[str, str | None]:
        """Decide raw direction and optional stop context for phrasing."""
        if center_obstacle >= self._center_obstacle_threshold:
            if (
                scores["left"] >= self._safe_threshold
                and scores["left"] > scores["right"] + self._side_margin * 0.5
            ):
                return "MOVE LEFT", "obstacle"
            if (
                scores["right"] >= self._safe_threshold
                and scores["right"] > scores["left"] + self._side_margin * 0.5
            ):
                return "MOVE RIGHT", "obstacle"
            return "STOP", "obstacle"

        if all(v < self._safe_threshold for v in scores.values()):
            return "STOP", "blocked"

        best_region = max(scores, key=scores.get)  # type: ignore[arg-type]

        if best_region in ("left", "right"):
            if scores[best_region] - scores["center"] < self._side_margin:
                return "MOVE FORWARD", None

        return {
            "left": "MOVE LEFT",
            "center": "MOVE FORWARD",
            "right": "MOVE RIGHT",
        }[best_region], None

    def _apply_hysteresis(self, direction: str, scores: dict[str, float]) -> str:
        """Stay on forward until a side is clearly better."""
        if self._last_smoothed != "MOVE FORWARD":
            return direction

        margin = self._side_margin + self._hysteresis_extra
        if direction == "MOVE LEFT" and scores["left"] - scores["center"] < margin:
            return "MOVE FORWARD"
        if direction == "MOVE RIGHT" and scores["right"] - scores["center"] < margin:
            return "MOVE FORWARD"
        return direction

    def _smooth(self) -> str:
        """Majority vote over the history buffer."""
        if not self._history:
            return "STOP"
        counter = Counter(self._history)
        return counter.most_common(1)[0][0]

    def update(self, class_map: np.ndarray, id2label: dict) -> dict:
        """Process one frame and return direction, phrase, and debug info."""
        self._ensure_class_ids(id2label)
        scores = self._compute_scores(class_map)
        center_obstacle = self._center_obstacle_ratio(class_map)

        raw, stop_reason = self._pick_direction(scores, center_obstacle)
        self._history.append(raw)
        smoothed = self._apply_hysteresis(self._smooth(), scores)
        self._last_smoothed = smoothed

        region = DIRECTION_TO_REGION[smoothed]
        if smoothed == "STOP":
            display_reason = (
                "obstacle"
                if center_obstacle >= self._center_obstacle_threshold
                else "blocked"
            )
            phrase = phrase_for("STOP", display_reason)
        else:
            display_reason = None
            phrase = phrase_for(smoothed)

        return {
            "direction": smoothed,
            "raw_direction": raw,
            "phrase": phrase,
            "region": region,
            "stop_reason": display_reason,
            "scores": scores,
            "center_obstacle": center_obstacle,
        }
