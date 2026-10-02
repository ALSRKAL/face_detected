"""Smoothed FPS measurement."""

from __future__ import annotations

import time


class FPSMeter:
    """Exponentially smoothed frames-per-second meter.

    The naive ``1 / (now - last)`` from the original script jumped around by
    tens of FPS frame to frame; an EMA over inter-frame deltas gives a number
    a human can actually read.
    """

    def __init__(self, history: int = 30, smoothing: float = 0.9) -> None:
        if history < 1:
            raise ValueError("history must be >= 1")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be within [0.0, 1.0)")
        self.history = history
        self.smoothing = smoothing
        self._last_time: float | None = None
        self._avg_delta: float | None = None
        self._frames = 0

    def tick(self, now: float | None = None) -> float:
        """Register one frame; returns the current smoothed FPS."""
        now = time.monotonic() if now is None else now
        if self._last_time is not None:
            delta = max(now - self._last_time, 1e-9)
            if self._avg_delta is None:
                self._avg_delta = delta
            else:
                self._avg_delta = (
                    self.smoothing * self._avg_delta + (1.0 - self.smoothing) * delta
                )
        self._last_time = now
        self._frames += 1
        return self.fps

    @property
    def fps(self) -> float:
        if self._avg_delta is None:
            return 0.0
        return 1.0 / self._avg_delta

    @property
    def frame_count(self) -> int:
        return self._frames

    def reset(self) -> None:
        self._last_time = None
        self._avg_delta = None
        self._frames = 0
