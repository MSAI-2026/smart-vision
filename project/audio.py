"""
audio.py -- Non-blocking TTS announcements for navigation directions.

Uses pyttsx3 in a dedicated daemon thread so speech never blocks the
camera loop.  Only announces when the direction actually changes.
"""

import queue
import threading

import pyttsx3


class AudioNavigator:
    """Background TTS that speaks navigation directions on change."""

    def __init__(self) -> None:
        self._last_announced: str | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Worker loop: owns the pyttsx3 engine and processes messages."""
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)

        while True:
            msg = self._queue.get()
            if msg is None:
                break

            # Drain any stale messages so we always speak the latest.
            latest = msg
            while not self._queue.empty():
                try:
                    newer = self._queue.get_nowait()
                    if newer is None:
                        engine.stop()
                        return
                    latest = newer
                except queue.Empty:
                    break

            engine.say(latest)
            engine.runAndWait()

    def announce(self, direction: str) -> None:
        """Queue a direction for speaking, only if it changed."""
        if direction == self._last_announced:
            return
        self._last_announced = direction
        self._queue.put(direction)

    def shutdown(self) -> None:
        """Signal the background thread to exit."""
        self._queue.put(None)
        self._thread.join(timeout=2.0)
