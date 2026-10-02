"""Mouth Aspect Ratio (MAR) based open/closed mouth detection."""

from __future__ import annotations

import math

from facedetected.config import MouthConfig

# Small hysteresis so the indicator does not flicker around the threshold.
_CLOSE_RATIO = 0.8


def mouth_aspect_ratio(landmarks) -> float | None:
    """MAR = inner-lip gap / inter-corner width (``None`` when degenerate)."""
    try:
        left = landmarks[61]
        right = landmarks[291]
        top = landmarks[13]
        bottom = landmarks[14]
    except IndexError:
        return None

    width = math.dist((left.x, left.y), (right.x, right.y))
    if width < 1e-6:
        return None
    return math.dist((top.x, top.y), (bottom.x, bottom.y)) / width


class MouthDetector:
    """Per-face mouth-state tracker with light hysteresis."""

    def __init__(self, cfg: MouthConfig | None = None) -> None:
        self.cfg = cfg or MouthConfig()
        self.is_open = False

    def update(self, mar: float | None) -> bool:
        if mar is None:
            return self.is_open
        open_threshold = self.cfg.mar_open_threshold
        close_threshold = open_threshold * _CLOSE_RATIO
        if self.is_open:
            if mar < close_threshold:
                self.is_open = False
        elif mar > open_threshold:
            self.is_open = True
        return self.is_open

    def reset(self) -> None:
        self.is_open = False
