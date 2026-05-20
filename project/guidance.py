"""
guidance.py -- Shared navigation labels and spoken phrases.

Single source of truth for on-screen text and TTS so sight and sound match.
"""

DIRECTIONS = ("MOVE LEFT", "MOVE FORWARD", "MOVE RIGHT", "STOP")

DIRECTION_TO_REGION: dict[str, str | None] = {
    "MOVE LEFT": "left",
    "MOVE FORWARD": "center",
    "MOVE RIGHT": "right",
    "STOP": None,
}

PHRASES: dict[str, str] = {
    "MOVE LEFT": "Path on your left",
    "MOVE FORWARD": "Go ahead",
    "MOVE RIGHT": "Path on your right",
    "STOP": "Stop",
}

STOP_PHRASES: dict[str, str] = {
    "blocked": "Path blocked ahead",
    "obstacle": "Obstacle ahead",
}


def phrase_for(direction: str, stop_reason: str | None = None) -> str:
    """Return the user-facing phrase for a direction and optional stop context."""
    if direction == "STOP" and stop_reason:
        return STOP_PHRASES.get(stop_reason, PHRASES["STOP"])
    return PHRASES.get(direction, direction)
