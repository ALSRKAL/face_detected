"""Canonical MediaPipe face-mesh landmark indices used by the analytics.

These indices are stable across the MediaPipe face_landmarker model
(478 points, the last 10 being the refined iris rings).
"""

from __future__ import annotations

# Eye Aspect Ratio sextets: (p1, p2, p3, p4, p5, p6)
# EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
LEFT_EYE_EAR: tuple[int, int, int, int, int, int] = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR: tuple[int, int, int, int, int, int] = (362, 385, 387, 263, 373, 380)

# Iris centres (refined landmarks) — handy for gaze work.
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Head-pose sextet: nose tip, chin, left eye outer, right eye outer,
# left mouth corner, right mouth corner.
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_CENTER = 159
RIGHT_EYE_CENTER = 386

MOUTH_CORNERS: tuple[int, int] = (61, 291)
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

POSE_SEXTET: tuple[int, int, int, int, int, int] = (
    NOSE_TIP,
    CHIN,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    61,
    291,
)
