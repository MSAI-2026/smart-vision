"""
audio.py -- Non-blocking TTS announcements for navigation directions.

Uses pyttsx3 in a dedicated daemon thread so speech never blocks the
camera loop.  Announces only after a direction is stable for several
frames, with a cooldown between utterances.  Interrupts in-progress
speech when a newer phrase is queued.
"""

import queue
import threading
import time

import pyttsx3

# Prefer these voice name fragments (macOS / cross-platform)
_PREFERRED_VOICES = ("Samantha", "Karen", "Daniel", "Moira", "en_US", "en_GB")


class AudioNavigator:
    """Background TTS that speaks navigation phrases on direction change."""

    def __init__(
        self,
        *,
        cooldown_s: float = 2.5,
        stable_frames: int = 3,
        stop_stable_frames: int = 6,
    ) -> None:
        self._cooldown_s = cooldown_s
        self._stable_frames = stable_frames
        self._stop_stable_frames = stop_stable_frames

        self._last_announced: str | None = None
        self._last_spoke_at: float = 0.0
        self._pending: str | None = None
        self._pending_count: int = 0

        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _configure_engine(engine: pyttsx3.Engine) -> None:
        """Pick a clear voice and speaking rate."""
        engine.setProperty("rate", 175)
        engine.setProperty("volume", 1.0)

        voices = engine.getProperty("voices") or []
        for fragment in _PREFERRED_VOICES:
            for voice in voices:
                name = getattr(voice, "name", "") or ""
                vid = getattr(voice, "id", "") or ""
                if fragment in name or fragment in vid:
                    engine.setProperty("voice", voice.id)
                    return

    def _run(self) -> None:
        """Worker loop: owns the pyttsx3 engine and processes messages."""
        engine = pyttsx3.init()
        self._configure_engine(engine)

        while True:
            msg = self._queue.get()
            if msg is None:
                break

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

            engine.stop()
            engine.say(latest)
            engine.runAndWait()

    def announce(self, direction: str, phrase: str) -> None:
        """Queue speech after direction is stable and cooldown has elapsed."""
        if direction == self._last_announced:
            self._pending = None
            self._pending_count = 0
            return

        if direction != self._pending:
            self._pending = direction
            self._pending_count = 1
        else:
            self._pending_count += 1

        required = (
            self._stop_stable_frames
            if direction == "STOP"
            else self._stable_frames
        )
        if self._pending_count < required:
            return

        now = time.monotonic()
        if now - self._last_spoke_at < self._cooldown_s:
            return

        self._last_announced = direction
        self._pending = None
        self._pending_count = 0
        self._last_spoke_at = now
        self._queue.put(phrase)

    def shutdown(self) -> None:
        """Signal the background thread to exit."""
        self._queue.put(None)
        self._thread.join(timeout=2.0)
