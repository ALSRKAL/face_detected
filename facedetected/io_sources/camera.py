"""Webcam source with automatic reconnection."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from facedetected.config import SourceConfig
from facedetected.io_sources.base import Frame, FrameSource

logger = logging.getLogger(__name__)


class CameraSource(FrameSource):
    """Read frames from a local camera, surviving transient device hiccups.

    ``max_reconnect_attempts=0`` (the default) retries forever, which is what
    an unattended monitor wants; set a positive number to give up eventually.
    """

    def __init__(self, config: SourceConfig | None = None, index: int | None = None) -> None:
        cfg = config or SourceConfig()
        super().__init__(name=f"camera:{index if index is not None else cfg.camera_index}")
        self.config = cfg
        self._index = index if index is not None else cfg.camera_index
        self._capture: cv2.VideoCapture | None = None
        self._frame_index = 0
        self._opened_at = time.monotonic()

    @property
    def fps(self) -> float:
        value = 0.0
        if self._capture is not None:
            value = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return value if value > 1.0 else 30.0

    def open(self) -> None:
        if self._capture is not None:
            return
        self._capture = self._connect()
        self._open = True
        self._opened_at = time.monotonic()
        self._frame_index = 0
        logger.info("Camera %s opened", self._name)

    def read(self) -> Frame | None:
        if self._capture is None:
            raise RuntimeError("camera not opened; call open() or use a with-block")

        ok, image = self._capture.read()
        if not ok:
            if not self._reconnect():
                return None
            ok, image = self._capture.read()
            if not ok:
                return None

        image = self._maybe_resize(image)
        elapsed_ms = int((time.monotonic() - self._opened_at) * 1000)
        frame = Frame(image=image, index=self._frame_index, timestamp_ms=elapsed_ms)
        self._frame_index += 1
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            self._open = False
            logger.info("Camera %s closed", self._name)

    # -- internals -----------------------------------------------------------

    def _connect(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self._index, cv2.CAP_ANY)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"cannot open camera {self._index}")
        if self.config.target_width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.target_width)
        if self.config.target_height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.target_height)
        return capture

    def _reconnect(self) -> bool:
        attempts = 0
        while True:
            attempts += 1
            logger.warning(
                "camera %s dropped a frame; reconnecting (attempt %d)", self._name, attempts
            )
            self._capture.release()
            time.sleep(self.config.reconnect_delay_s)
            try:
                self._capture = self._connect()
            except RuntimeError:
                pass
            if self._capture is not None and self._capture.isOpened():
                ok, _ = self._capture.read()
                if ok:
                    logger.info("camera %s reconnected", self._name)
                    return True
            if self.config.max_reconnect_attempts and attempts >= self.config.max_reconnect_attempts:
                logger.error("camera %s reconnect failed %d times; giving up", self._name, attempts)
                return False

    def _maybe_resize(self, image: np.ndarray) -> np.ndarray:
        if not self.config.target_width or not self.config.target_height:
            return image
        h, w = image.shape[:2]
        if (w, h) == (self.config.target_width, self.config.target_height):
            return image
        return cv2.resize(
            image,
            (self.config.target_width, self.config.target_height),
            interpolation=cv2.INTER_AREA,
        )
