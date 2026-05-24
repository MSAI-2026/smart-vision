"""
navigation.py -- Obstacle-avoidance direction from a segmentation map.

Per-frame pipeline:

    walkable mask
        -> largest connected free-space region
        -> centroid steering
        -> emergency stop zone (overrides smoothing when triggered)

Scores in the lower path band support sidestep and hysteresis; center-column
obstacle checks can override centroid steering before the final direction.
"""

from collections import Counter, deque

import cv2
import numpy as np

from project.guidance import DIRECTION_TO_REGION, phrase_for
from project.overlay import CATEGORIES, _resolve_ids

_REGION_TO_DIRECTION = {
    "left": "MOVE LEFT",
    "center": "MOVE FORWARD",
    "right": "MOVE RIGHT",
}


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
        danger_height_ratio: float = 0.2,
        danger_width_ratio: float = 0.2,
        danger_obstacle_threshold: float = 0.15,
        weight_max: float = 2.5,
        obstacle_history_len: int = 5,
        growth_delta_threshold: float = 0.06,
        growth_stop_threshold: float = 0.15,
        min_component_area_ratio: float = 0.005,
    ):
        self._history_len = history_len
        self._safe_threshold = safe_threshold
        self._side_margin = side_margin
        self._hysteresis_extra = hysteresis_extra
        self._center_obstacle_threshold = center_obstacle_threshold
        self._path_top_ratio = path_top_ratio
        self._danger_height_ratio = danger_height_ratio
        self._danger_width_ratio = danger_width_ratio
        self._danger_obstacle_threshold = danger_obstacle_threshold
        self._weight_max = weight_max
        self._growth_delta_threshold = growth_delta_threshold
        self._growth_stop_threshold = growth_stop_threshold
        self._min_component_area_ratio = min_component_area_ratio

        self._history: deque[str] = deque(maxlen=history_len)
        self._obstacle_history: deque[float] = deque(maxlen=obstacle_history_len)
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

    def _danger_slice(self, h: int, w: int) -> tuple[slice, slice]:
        """Bottom-center band for emergency collision detection."""
        row_sl = slice(int(h * (1.0 - self._danger_height_ratio)), h)
        half_w = self._danger_width_ratio / 2.0
        col_sl = slice(int(w * (0.5 - half_w)), int(w * (0.5 + half_w)))
        return row_sl, col_sl

    def _region_slices(self, h: int, w: int) -> dict[str, tuple[slice, slice]]:
        third = w // 3
        rows = self._path_slice(h)
        return {
            "left": (rows, slice(0, third)),
            "center": (rows, slice(third, 2 * third)),
            "right": (rows, slice(2 * third, w)),
        }

    def _region_from_x(self, x: float, w: int) -> str:
        third = w // 3
        if x < third:
            return "left"
        if x < 2 * third:
            return "center"
        return "right"

    def _walkable_mask(self, class_map: np.ndarray) -> np.ndarray:
        """Binary mask of walkable class pixels."""
        return np.isin(class_map, list(self._walk_ids))

    def _path_weights(self, n_rows: int) -> np.ndarray:
        """Heavier weights toward the bottom of the path band."""
        if n_rows <= 0:
            return np.ones((1, 1), dtype=np.float64)
        return np.linspace(1.0, self._weight_max, n_rows, dtype=np.float64).reshape(-1, 1)

    def _compute_scores(self, class_map: np.ndarray) -> dict[str, float]:
        """Bottom-weighted walkable score per column in the lower path band."""
        h, w = class_map.shape[:2]
        safe_mask = self._walkable_mask(class_map)
        row_sl = self._path_slice(h)
        n_rows = h - row_sl.start
        weights = self._path_weights(n_rows)
        scores: dict[str, float] = {}

        for name, (rows, col_sl) in self._region_slices(h, w).items():
            region_safe = safe_mask[rows, col_sl].astype(np.float64)
            w_region = np.broadcast_to(weights, region_safe.shape)
            scores[name] = float((region_safe * w_region).sum()) / float(w_region.sum())

        return scores

    def _danger_obstacle_ratio(self, class_map: np.ndarray) -> float:
        """Share of obstacle pixels in the bottom-center danger zone."""
        h, w = class_map.shape[:2]
        row_sl, col_sl = self._danger_slice(h, w)
        obstacle_mask = ~np.isin(class_map, list(self._passable_ids))
        region = obstacle_mask[row_sl, col_sl]
        return float(region.sum()) / max(region.size, 1)

    def _center_obstacle_ratio(self, class_map: np.ndarray) -> float:
        """Share of obstacle pixels in the lower center column."""
        h, w = class_map.shape[:2]
        row_sl, col_sl = self._region_slices(h, w)["center"]
        obstacle_mask = ~np.isin(class_map, list(self._passable_ids))
        region = obstacle_mask[row_sl, col_sl]
        return float(region.sum()) / max(region.size, 1)

    def _largest_free_region(
        self, class_map: np.ndarray, walk_mask: np.ndarray,
    ) -> tuple[str | None, tuple[float, float] | None, int]:
        """
        Largest connected walkable component in the path band.

        Returns (region_name, centroid_xy in frame coords, pixel_area).
        """
        h, w = class_map.shape[:2]
        row_sl = self._path_slice(h)
        band = walk_mask[row_sl, :].astype(np.uint8)

        _, _, stats, centroids = cv2.connectedComponentsWithStats(
            band, connectivity=8,
        )
        min_area = band.size * self._min_component_area_ratio

        best_label = -1
        best_area = 0
        for label in range(1, len(stats)):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area and area > best_area:
                best_area = area
                best_label = label

        if best_label < 0:
            return None, None, 0

        cx = float(centroids[best_label][0])
        cy = float(row_sl.start + centroids[best_label][1])
        region = self._region_from_x(cx, w)
        return region, (cx, cy), best_area

    def _obstacle_growing(self, current: float) -> bool:
        """True when center obstacle ratio is rising quickly vs recent frames."""
        if len(self._obstacle_history) < 3:
            return False
        prev_mean = sum(self._obstacle_history) / len(self._obstacle_history)
        return current - prev_mean >= self._growth_delta_threshold

    def _try_sidestep(self, scores: dict[str, float]) -> tuple[str, str] | None:
        """Prefer a clear side path when the center is blocked."""
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
        return None

    def _centroid_steering(
        self, centroid_region: str | None, scores: dict[str, float],
    ) -> tuple[str, str | None]:
        """Steer toward the largest free-space centroid's column."""
        if centroid_region is None:
            return "STOP", "blocked"
        return self._direction_for_region(centroid_region, scores)

    def _direction_for_region(
        self, region: str, scores: dict[str, float],
    ) -> tuple[str, str | None]:
        """Map a target region to a direction, with forward bias on weak sides."""
        if region in ("left", "right"):
            if scores[region] - scores["center"] < self._side_margin:
                return "MOVE FORWARD", None
        return _REGION_TO_DIRECTION[region], None

    def _pick_direction(
        self,
        scores: dict[str, float],
        center_obstacle: float,
        centroid_region: str | None,
        obstacle_growing: bool,
    ) -> tuple[str, str | None]:
        """Decide raw direction and optional stop context for phrasing."""
        if center_obstacle >= self._center_obstacle_threshold:
            sidestep = self._try_sidestep(scores)
            if sidestep:
                return sidestep
            return "STOP", "obstacle"

        if obstacle_growing:
            if center_obstacle >= self._growth_stop_threshold:
                return "STOP", "obstacle"
            sidestep = self._try_sidestep(scores)
            if sidestep:
                return sidestep

        if all(v < self._safe_threshold for v in scores.values()):
            return "STOP", "blocked"

        return self._centroid_steering(centroid_region, scores)

    def _emergency_stop(self, class_map: np.ndarray) -> bool:
        """True when the bottom-center danger zone has too many obstacles."""
        return (
            self._danger_obstacle_ratio(class_map)
            >= self._danger_obstacle_threshold
        )

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

    def _build_result(
        self,
        *,
        direction: str,
        raw_direction: str,
        scores: dict[str, float],
        center_obstacle: float,
        stop_reason: str | None,
        safe_centroid: tuple[float, float] | None,
        largest_safe_area: int,
        emergency_stop: bool,
    ) -> dict:
        region = DIRECTION_TO_REGION[direction]
        if direction == "STOP":
            display_reason = stop_reason or (
                "obstacle"
                if center_obstacle >= self._center_obstacle_threshold
                else "blocked"
            )
            phrase = phrase_for("STOP", display_reason)
        else:
            display_reason = None
            phrase = phrase_for(direction)

        return {
            "direction": direction,
            "raw_direction": raw_direction,
            "phrase": phrase,
            "region": region,
            "stop_reason": display_reason,
            "scores": scores,
            "center_obstacle": center_obstacle,
            "emergency_stop": emergency_stop,
            "safe_centroid": safe_centroid,
            "largest_safe_area": largest_safe_area,
        }

    def update(self, class_map: np.ndarray, id2label: dict) -> dict:
        """Process one frame and return direction, phrase, and debug info."""
        self._ensure_class_ids(id2label)

        walk_mask = self._walkable_mask(class_map)
        centroid_region, safe_centroid, largest_safe_area = self._largest_free_region(
            class_map, walk_mask,
        )

        scores = self._compute_scores(class_map)
        center_obstacle = self._center_obstacle_ratio(class_map)
        obstacle_growing = self._obstacle_growing(center_obstacle)
        self._obstacle_history.append(center_obstacle)

        if self._emergency_stop(class_map):
            self._last_smoothed = "STOP"
            return self._build_result(
                direction="STOP",
                raw_direction="STOP",
                scores=scores,
                center_obstacle=center_obstacle,
                stop_reason="obstacle",
                safe_centroid=safe_centroid,
                largest_safe_area=largest_safe_area,
                emergency_stop=True,
            )

        raw, stop_reason = self._pick_direction(
            scores, center_obstacle, centroid_region, obstacle_growing,
        )
        self._history.append(raw)
        smoothed = self._apply_hysteresis(self._smooth(), scores)
        self._last_smoothed = smoothed

        return self._build_result(
            direction=smoothed,
            raw_direction=raw,
            scores=scores,
            center_obstacle=center_obstacle,
            stop_reason=stop_reason if smoothed == "STOP" else None,
            safe_centroid=safe_centroid,
            largest_safe_area=largest_safe_area,
            emergency_stop=False,
        )
