"""
display.py -- Visual navigation hints on the camera overlay.

Draws path columns, highlights the recommended band, and a direction arrow
that matches the spoken guidance phrase.
"""

import cv2
import numpy as np

from project.guidance import DIRECTION_TO_REGION

# BGR
_GUIDE_COLOUR = (160, 160, 160)
_HIGHLIGHT_COLOUR = (0, 220, 0)
_ARROW_COLOUR = (0, 255, 255)
_STOP_COLOUR = (0, 0, 255)
_DANGER_COLOUR = (0, 0, 220)
_CENTROID_COLOUR = (255, 200, 0)

_PATH_TOP_RATIO = 0.45
_DANGER_HEIGHT_RATIO = 0.2
_DANGER_WIDTH_RATIO = 0.2


def _highlight_region(frame: np.ndarray, region: str) -> None:
    """Tint the active path column (lower frame only)."""
    h, w = frame.shape[:2]
    third = w // 3
    path_top = int(h * _PATH_TOP_RATIO)

    if region == "left":
        x0, x1 = 0, third
    elif region == "center":
        x0, x1 = third, 2 * third
    else:
        x0, x1 = 2 * third, w

    tint = frame.copy()
    cv2.rectangle(tint, (x0, path_top), (x1, h), _HIGHLIGHT_COLOUR, -1)
    cv2.addWeighted(tint, 0.18, frame, 0.82, 0, frame)


def _danger_rect(w: int, h: int) -> tuple[int, int, int, int]:
    """Bottom-center band matching NavigationAdvisor._danger_slice."""
    y0 = int(h * (1.0 - _DANGER_HEIGHT_RATIO))
    half_w = _DANGER_WIDTH_RATIO / 2.0
    x0 = int(w * (0.5 - half_w))
    x1 = int(w * (0.5 + half_w))
    return x0, y0, x1, h


def _draw_danger_zone(frame: np.ndarray, nav: dict) -> None:
    """Outline the emergency stop zone; fill when emergency_stop is active."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = _danger_rect(w, h)
    if nav.get("emergency_stop"):
        tint = frame.copy()
        cv2.rectangle(tint, (x0, y0), (x1, y1), _DANGER_COLOUR, -1)
        cv2.addWeighted(tint, 0.35, frame, 0.65, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), _DANGER_COLOUR, 2, cv2.LINE_AA)


def _draw_centroid(frame: np.ndarray, nav: dict) -> None:
    """Mark the largest free-space region centroid used for steering."""
    centroid = nav.get("safe_centroid")
    if not centroid:
        return
    cx, cy = int(centroid[0]), int(centroid[1])
    cv2.circle(frame, (cx, cy), 10, _CENTROID_COLOUR, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, _CENTROID_COLOUR, -1, cv2.LINE_AA)


def _draw_arrow(frame: np.ndarray, direction: str) -> None:
    """Large arrow at the bottom indicating movement."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h - 50
    length = min(w, h) // 6

    if direction == "MOVE FORWARD":
        tip = (cx, cy - length)
    elif direction == "MOVE LEFT":
        tip = (cx - length, cy)
    elif direction == "MOVE RIGHT":
        tip = (cx + length, cy)
    else:
        cv2.circle(frame, (cx, cy), 28, _STOP_COLOUR, 3, cv2.LINE_AA)
        cv2.line(frame, (cx - 18, cy - 18), (cx + 18, cy + 18), _STOP_COLOUR, 3, cv2.LINE_AA)
        cv2.line(frame, (cx + 18, cy - 18), (cx - 18, cy + 18), _STOP_COLOUR, 3, cv2.LINE_AA)
        return

    cv2.arrowedLine(
        frame, (cx, cy), tip, _ARROW_COLOUR, 4, tipLength=0.35, line_type=cv2.LINE_AA,
    )


def draw_navigation_hints(frame: np.ndarray, nav: dict) -> None:
    """Draw column guides, active-region highlight, phrase, and arrow."""
    h, w = frame.shape[:2]
    third = w // 3

    cv2.line(frame, (third, 0), (third, h), _GUIDE_COLOUR, 1, cv2.LINE_AA)
    cv2.line(frame, (2 * third, 0), (2 * third, h), _GUIDE_COLOUR, 1, cv2.LINE_AA)

    _draw_danger_zone(frame, nav)
    _draw_centroid(frame, nav)

    region = nav.get("region")
    if region:
        _highlight_region(frame, region)

    _draw_arrow(frame, nav["direction"])

    phrase = nav.get("phrase", nav["direction"])
    cv2.putText(
        frame, phrase,
        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
        (255, 255, 255), 2, cv2.LINE_AA,
    )

    scores = nav["scores"]
    score_text = (
        f"L:{scores['left']:.0%}  C:{scores['center']:.0%}  R:{scores['right']:.0%}"
    )
    cv2.putText(
        frame, score_text,
        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (200, 200, 200), 1, cv2.LINE_AA,
    )
