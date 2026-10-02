"""Manual OpenCV drawing of mesh, contours, boxes and labels.

MediaPipe 1.0 dropped ``mp.solutions.drawing_utils``; drawing here keeps full
control of colours and cost, and lets us highlight analytic regions later.
Connection index tables come from the official FaceLandmarksConnections.
"""

from __future__ import annotations

import cv2

from mediapipe.tasks.python.vision import FaceLandmarksConnections as _C

from facedetected.config import OverlayConfig
from facedetected.engine.results import DetectionResult, Face


def _pairs(constant) -> tuple[tuple[int, int], ...]:
    return tuple((c.start, c.end) for c in constant)


TESSELATION = _pairs(_C.FACE_LANDMARKS_TESSELATION)
CONTOURS = _pairs(_C.FACE_LANDMARKS_CONTOURS)
LEFT_EYE = _pairs(_C.FACE_LANDMARKS_LEFT_EYE)
RIGHT_EYE = _pairs(_C.FACE_LANDMARKS_RIGHT_EYE)
LEFT_IRIS = _pairs(_C.FACE_LANDMARKS_LEFT_IRIS)
RIGHT_IRIS = _pairs(_C.FACE_LANDMARKS_RIGHT_IRIS)
LIPS = _pairs(_C.FACE_LANDMARKS_LIPS)


def _line_pairs(img, face: Face, pairs, color, thickness: int) -> None:
    lms = face.landmarks
    for start, end in pairs:
        a, b = lms[start], lms[end]
        cv2.line(img, (a.ix, a.iy), (b.ix, b.iy), color, thickness, cv2.LINE_AA)


def draw_mesh(img, face: Face, cfg: OverlayConfig, *, tesselation: bool = False) -> None:
    """Draw the face mesh; with ``tesselation`` the full 2556-edge web."""
    _line_pairs(img, face, TESSELATION if tesselation else CONTOURS,
                cfg.mesh_color_bgr, cfg.mesh_thickness)


def draw_contours(img, face: Face, cfg: OverlayConfig) -> None:
    """Draw the eyes / brows / nose / lips / oval contour set."""
    _line_pairs(img, face, CONTOURS, cfg.mesh_color_bgr, cfg.mesh_thickness)


def draw_iris(img, face: Face, cfg: OverlayConfig) -> None:
    """Highlight both iris rings in the accent colour."""
    _line_pairs(img, face, LEFT_IRIS, cfg.accent_color_bgr, cfg.mesh_thickness + 1)
    _line_pairs(img, face, RIGHT_IRIS, cfg.accent_color_bgr, cfg.mesh_thickness + 1)


def draw_box(img, face: Face, cfg: OverlayConfig) -> None:
    """Bounding box plus a small 'face #i [score]' label."""
    box = face.box
    cv2.rectangle(img, box.top_left, (box.x2, box.y2), cfg.box_color_bgr, cfg.box_thickness)
    score = f" {face.score * 100:.0f}%" if face.score is not None else ""
    label = f"face {face.index}{score}"
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    x, y = box.x, max(box.y - 6, th + 2)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + baseline - 2),
                  cfg.box_color_bgr, cv2.FILLED)
    cv2.putText(img, label, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_landmark_ids(img, face: Face, cfg: OverlayConfig) -> None:
    """Draw the numeric index of every landmark (debug view)."""
    for i, lm in enumerate(face.landmarks):
        cv2.putText(img, str(i), (lm.ix, lm.iy), cv2.FONT_HERSHEY_PLAIN,
                    0.5, cfg.text_color_bgr, 1, cv2.LINE_AA)


def draw_result(
    img,
    result: DetectionResult,
    cfg: OverlayConfig,
    *,
    show_scores: bool = True,
) -> None:
    """Convenience: render a whole DetectionResult according to cfg."""
    for face in result.faces:
        if cfg.draw_mesh:
            draw_contours(img, face, cfg)
            if cfg.draw_tesselation:
                draw_mesh(img, face, cfg, tesselation=True)
        if cfg.draw_iris:
            draw_iris(img, face, cfg)
        if cfg.show_landmark_ids:
            draw_landmark_ids(img, face, cfg)
        if cfg.draw_boxes:
            draw_box(img, face, cfg)
