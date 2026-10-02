"""Still-image source: yields the image once, then EOF."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from facedetected.io_sources.base import Frame, FrameSource

logger = logging.getLogger(__name__)


class ImageSource(FrameSource):
    """Single still image, usable as a one-frame source (tests, CLI, demos)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        super().__init__(name=str(self.path))
        self._image: cv2.Mat | None = None
        self._emitted = False

    def open(self) -> None:
        if self._image is not None:
            return
        if not self.path.is_file():
            raise FileNotFoundError(f"image not found: {self.path}")
        self._image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if self._image is None:
            raise RuntimeError(f"cannot decode image: {self.path}")
        self._open = True
        self._emitted = False
        logger.info("Image %s loaded (%dx%d)", self.path, self._image.shape[1], self._image.shape[0])

    def read(self) -> Frame | None:
        if self._image is None:
            raise RuntimeError("image source not opened; call open() or use a with-block")
        if self._emitted:
            return None
        self._emitted = True
        return Frame(image=self._image.copy(), index=0, timestamp_ms=0)

    def close(self) -> None:
        self._image = None
        self._open = False
        self._emitted = False
