"""FaceLandmarker engine: 478-point dense face mesh via MediaPipe Tasks."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from facedetected.config import DetectorConfig
from facedetected.engine.results import DetectionResult, Face, FaceBox, Landmark
from facedetected.models.manager import FACE_LANDMARKER, ensure_model

logger = logging.getLogger(__name__)

_RUNNING_MODES = {
    "image": "IMAGE",
    "video": "VIDEO",
    "live_stream": "LIVE_STREAM",
}


def _box_from_landmarks(landmarks: tuple[Landmark, ...], width: int, height: int) -> FaceBox:
    xs = [lm.ix for lm in landmarks]
    ys = [lm.iy for lm in landmarks]
    x0, x1 = max(min(xs), 0), min(max(xs), width - 1)
    y0, y1 = max(min(ys), 0), min(max(ys), height - 1)
    return FaceBox(x0, y0, max(x1 - x0, 1), max(y1 - y0, 1))


class FaceLandmarkerEngine:
    """Dense face-mesh detection with a clean, testable API.

    Example:
        with FaceLandmarkerEngine(DetectorConfig(), mode="image") as engine:
            result = engine.detect_image(bgr_frame)
            for face in result.faces:
                print(face.box, len(face.landmarks))
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
        *,
        mode: str = "video",
        models_dir: Path | None = None,
    ) -> None:
        if mode not in _RUNNING_MODES:
            raise ValueError(f"mode must be one of {sorted(_RUNNING_MODES)}, got {mode!r}")
        self.config = config or DetectorConfig()
        self.mode = mode
        self._models_dir = models_dir
        self._landmarker = None
        self._mp = None
        self._last_timestamp_ms = -1

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> FaceLandmarkerEngine:
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def open(self) -> None:
        if self._landmarker is not None:
            return
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        model_path = ensure_model(FACE_LANDMARKER, self._models_dir)
        running_mode = getattr(mp_vision.RunningMode, _RUNNING_MODES[self.mode])
        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_faces=self.config.num_faces,
            min_face_detection_confidence=self.config.min_face_detection_confidence,
            min_face_presence_confidence=self.config.min_face_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_face_blendshapes=self.config.output_blendshapes,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        logger.info("FaceLandmarker ready (mode=%s, num_faces=%d)", self.mode, self.config.num_faces)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    # -- inference -----------------------------------------------------------

    def _to_mp_image(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

    def detect_image(self, frame_bgr: np.ndarray) -> DetectionResult:
        """One-shot detection for still images (mode='image')."""
        self._require_open()
        h, w = frame_bgr.shape[:2]
        mp_image = self._to_mp_image(frame_bgr)
        raw = self._landmarker.detect(mp_image)
        return self._build_result(raw, w, h, timestamp_ms=0)

    def detect_video(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> DetectionResult:
        """Detection for consecutive video frames (mode='video').

        ``timestamp_ms`` may be omitted; a monotonic clock is used instead.
        MediaPipe requires strictly increasing timestamps.
        """
        self._require_open()
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        h, w = frame_bgr.shape[:2]
        mp_image = self._to_mp_image(frame_bgr)
        raw = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return self._build_result(raw, w, h, timestamp_ms)

    # -- internals -----------------------------------------------------------

    def _require_open(self) -> None:
        if self._landmarker is None:
            raise RuntimeError("engine not opened; call open() or use a with-block")

    def _build_result(self, raw, width: int, height: int, timestamp_ms: int) -> DetectionResult:
        faces: list[Face] = []
        per_face = raw.face_landmarks or []
        blendshape_sets = raw.face_blendshapes or []

        for index, lms in enumerate(per_face):
            landmarks = tuple(
                Landmark(
                    x=lm.x * width,
                    y=lm.y * height,
                    z=lm.z * width,
                    nx=lm.x,
                    ny=lm.y,
                    nz=lm.z,
                )
                for lm in lms
            )
            box = _box_from_landmarks(landmarks, width, height)
            blendshapes = None
            if index < len(blendshape_sets):
                blendshapes = {
                    category.category_name: float(category.score)
                    for category in blendshape_sets[index]
                }
            faces.append(Face(index=index, box=box, landmarks=landmarks, blendshapes=blendshapes))

        return DetectionResult(faces=tuple(faces), width=width, height=height, timestamp_ms=timestamp_ms)
