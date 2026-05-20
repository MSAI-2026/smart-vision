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


def _highlight_region(frame: np.ndarray, region: str) -> None:
    """Tint the active path column (lower frame only)."""
    h, w = frame.shape[:2]
    third = w // 3
    path_top = int(h * 0.45)

    if region == "left":
        x0, x1 = 0, third
    elif region == "center":
        x0, x1 = third, 2 * third
    else:
        x0, x1 = 2 * third, w

    tint = frame.copy()
    cv2.rectangle(tint, (x0, path_top), (x1, h), _HIGHLIGHT_COLOUR, -1)
    cv2.addWeighted(tint, 0.18, frame, 0.82, 0, frame)


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
