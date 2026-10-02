"""Head pose (yaw / pitch / roll) from six canonical mesh landmarks.

A classic 6-point PnP against a generic 3D face model gives angles that are
more than good enough for a live HUD.  ``None`` is returned whenever OpenCV
cannot solve, so callers can degrade gracefully.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from facedetected.analytics.landmarks import POSE_SEXTET
from facedetected.engine.results import Face

# Generic 3D face model (classic six-point layout, arbitrary units),
# expressed in OpenCV camera coordinates: x right, y DOWN, z forward
# (away from the camera).  Keeping the model in image-space axes avoids
# the 180-degree pitch flip the classic y-up variant produces.
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),           # nose tip
        (0.0, 330.0, 65.0),        # chin
        (-225.0, -170.0, 135.0),   # left eye outer corner
        (225.0, -170.0, 135.0),    # right eye outer corner
        (-150.0, 150.0, 125.0),    # left mouth corner
        (150.0, 150.0, 125.0),     # right mouth corner
    ],
    dtype=np.float64,
)


class HeadPoseEstimator:
    """Stateless per-frame estimator; one instance per face tracker."""

    def update(self, face: Face, image_width: int, image_height: int) -> tuple[float, float, float] | None:
        """Return ``(yaw, pitch, roll)`` in degrees, or ``None``.

        ``image_width`` / ``image_height`` are the FULL frame dimensions:
        the PnP camera matrix must describe the real capture optics, not the
        face crop, or the solution degenerates.
        """
        if not face.has_landmarks:
            return None

        focal = float(image_width)
        image_points = np.array(
            [(face.landmarks[i].x, face.landmarks[i].y) for i in POSE_SEXTET],
            dtype=np.float64,
        )
        camera_matrix = np.array(
            [
                [focal, 0.0, image_width / 2.0],
                [0.0, focal, image_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rotation_vector, _ = cv2.solvePnP(
            _MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
        # angles = rotations about x (pitch), y (yaw), z (roll)
        pitch, yaw, roll = angles
        if any(math.isnan(a) for a in (pitch, yaw, roll)):
            return None
        return yaw, pitch, roll


def format_pose(pose: tuple[float, float, float] | None) -> str | None:
    """Compact HUD string, e.g. ``y+12 p-04 r+02``."""
    if pose is None:
        return None
    yaw, pitch, roll = pose
    return f"y{yaw:+04.0f} p{pitch:+04.0f} r{roll:+04.0f}"
