"""Video-file source with optional looping (handy for demos)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from facedetected.io_sources.base import Frame, FrameSource

logger = logging.getLogger(__name__)


class VideoFileSource(FrameSource):
    """Read frames from a video file; ``loop=True`` restarts at EOF forever."""

    def __init__(self, path: str | Path, *, loop: bool = False) -> None:
        self.path = Path(path)
        super().__init__(name=str(self.path))
        self.loop = loop
        self._capture: cv2.VideoCapture | None = None
        self._frame_index = 0
        self._fps = 30.0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        if self._capture is None:
            return 0
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def open(self) -> None:
        if self._capture is not None:
            return
        if not self.path.is_file():
            raise FileNotFoundError(f"video file not found: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"cannot open video file: {self.path}")
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = fps if fps > 1.0 else 30.0
        self._frame_index = 0
        self._open = True
        logger.info("Video %s opened (%.1f fps)", self.path, self._fps)

    def read(self) -> Frame | None:
        if self._capture is None:
            raise RuntimeError("video source not opened; call open() or use a with-block")

        ok, image = self._capture.read()
        if not ok:
            if not self.loop:
                return None
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self._capture.read()
            if not ok:
                return None

        timestamp_ms = int(self._frame_index * 1000 / self._fps)
        frame = Frame(image=image, index=self._frame_index, timestamp_ms=timestamp_ms)
        self._frame_index += 1
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            self._open = False
            logger.info("Video %s closed", self.path)
