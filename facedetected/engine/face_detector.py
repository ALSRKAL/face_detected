"""FaceDetector engine: fast bounding boxes via MediaPipe BlazeFace."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from facedetected.config import DetectorConfig
from facedetected.engine.results import DetectionResult, Face, FaceBox
from facedetected.models.manager import FACE_DETECTOR, ensure_model

logger = logging.getLogger(__name__)

_RUNNING_MODES = {
    "image": "IMAGE",
    "video": "VIDEO",
    "live_stream": "LIVE_STREAM",
}


class FaceDetectorEngine:
    """Lightweight face detection (bounding boxes + confidence) for when the
    dense mesh is unnecessary — e.g. counting people or pre-filtering frames.

    Example:
        with FaceDetectorEngine(mode="image") as engine:
            result = engine.detect_image(bgr_frame)
            print([f.box for f in result.faces])
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
        self._detector = None
        self._mp = None
        self._last_timestamp_ms = -1

    def __enter__(self) -> FaceDetectorEngine:
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def open(self) -> None:
        if self._detector is not None:
            return
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        model_path = ensure_model(FACE_DETECTOR, self._models_dir)
        running_mode = getattr(mp_vision.RunningMode, _RUNNING_MODES[self.mode])
        options = mp_vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            min_detection_confidence=self.config.min_face_detection_confidence,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)
        logger.info("FaceDetector ready (mode=%s)", self.mode)

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def detect_image(self, frame_bgr: np.ndarray) -> DetectionResult:
        """One-shot detection for still images (mode='image')."""
        self._require_open()
        h, w = frame_bgr.shape[:2]
        mp_image = self._to_mp_image(frame_bgr)
        raw = self._detector.detect(mp_image)
        return self._build_result(raw, w, h, timestamp_ms=0)

    def detect_video(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> DetectionResult:
        """Detection for consecutive video frames (mode='video')."""
        import time

        self._require_open()
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        h, w = frame_bgr.shape[:2]
        mp_image = self._to_mp_image(frame_bgr)
        raw = self._detector.detect_for_video(mp_image, timestamp_ms)
        return self._build_result(raw, w, h, timestamp_ms)

    # -- internals -----------------------------------------------------------

    def _require_open(self) -> None:
        if self._detector is None:
            raise RuntimeError("engine not opened; call open() or use a with-block")

    def _to_mp_image(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

    def _build_result(self, raw, width: int, height: int, timestamp_ms: int) -> DetectionResult:
        faces: list[Face] = []
        for index, detection in enumerate(raw.detections or []):
            bbox = detection.bounding_box
            score = None
            if detection.categories:
                score = float(detection.categories[0].score)
            box = FaceBox(
                x=max(0, int(bbox.origin_x)),
                y=max(0, int(bbox.origin_y)),
                w=max(1, int(bbox.width)),
                h=max(1, int(bbox.height)),
            ).clipped_to(width, height)
            faces.append(Face(index=index, box=box, landmarks=(), score=score))

        return DetectionResult(faces=tuple(faces), width=width, height=height, timestamp_ms=timestamp_ms)
