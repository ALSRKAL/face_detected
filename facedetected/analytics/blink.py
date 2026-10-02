"""Eye Aspect Ratio (EAR) blink detection with a hysteretic state machine."""

from __future__ import annotations

import math

from facedetected.config import BlinkConfig


def eye_aspect_ratio(landmarks, sextet: tuple[int, int, int, int, int, int]) -> float | None:
    """Compute EAR for one eye from six landmark indices.

    Returns ``None`` when the landmarks cannot form a valid ratio
    (degenerate geometry, zero inter-canthal distance, ...).
    """
    try:
        p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in sextet)
    except IndexError:
        return None

    horizontal = math.dist((p1.x, p1.y), (p4.x, p4.y))
    if horizontal < 1e-6:
        return None
    vertical = math.dist((p2.x, p2.y), (p6.x, p6.y)) + math.dist((p3.x, p3.y), (p5.x, p5.y))
    return vertical / (2.0 * horizontal)


def mean_ear(landmarks, left: tuple, right: tuple) -> float | None:
    """Average EAR over both eyes (more robust to one eye being occluded)."""
    left_ear = eye_aspect_ratio(landmarks, left)
    right_ear = eye_aspect_ratio(landmarks, right)
    values = [v for v in (left_ear, right_ear) if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


class BlinkDetector:
    """Per-face blink counter.

    Uses two thresholds (close / reopen) so sensor noise around one value
    cannot chatter the counter, and requires the eye to stay closed for
    ``min_closed_frames`` frames before a blink is registered.
    """

    def __init__(self, cfg: BlinkConfig | None = None) -> None:
        self.cfg = cfg or BlinkConfig()
        self.blink_count = 0
        self.is_closed = False
        self._closed_run = 0

    def update(self, ear: float | None) -> tuple[bool, bool]:
        """Feed one frame; returns ``(is_closed, blink_just_happened)``."""
        if ear is None:
            # No usable measurement: hold state, drop any partial run.
            self._closed_run = 0
            return self.is_closed, False

        blink = False
        if not self.is_closed:
            if ear < self.cfg.ear_closed_threshold:
                self._closed_run += 1
                if self._closed_run >= self.cfg.min_closed_frames:
                    self.is_closed = True
                    self._closed_run = 0
            else:
                self._closed_run = 0
        else:
            if ear > self.cfg.ear_open_threshold:
                self.is_closed = False
                self.blink_count += 1
                blink = True
        return self.is_closed, blink

    def reset(self) -> None:
        self.is_closed = False
        self._closed_run = 0
