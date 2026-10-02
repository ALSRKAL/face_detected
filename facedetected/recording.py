"""Annotated-stream recording via OpenCV VideoWriter."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from facedetected.config import SnapshotConfig

logger = logging.getLogger(__name__)

# (fourcc, extension) candidates in preference order; the first that opens wins.
_CODECS = (
    ("mp4v", "mp4"),
    ("avc1", "mp4"),
    ("XVID", "avi"),
    ("MJPG", "avi"),
)


class Recorder:
    """Lazily-created video writer with codec negotiation and clean stop."""

    def __init__(self, cfg: SnapshotConfig | None = None) -> None:
        self.cfg = cfg or SnapshotConfig()
        self._writer: cv2.VideoWriter | None = None
        self._path: Path | None = None
        self._frames_written = 0

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def start(self, fps: float, size: tuple[int, int]) -> Path:
        """Start a new recording; returns the target file path."""
        if self._writer is not None:
            raise RuntimeError("already recording")
        if fps <= 0:
            fps = 30.0

        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = self.cfg.directory / f"{self.cfg.prefix}_{stamp}_recording"
        self.cfg.directory.mkdir(parents=True, exist_ok=True)

        last_error = "no codec attempted"
        for fourcc_name, ext in _CODECS:
            path = base.with_suffix(f".{ext}")
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            writer = cv2.VideoWriter(str(path), fourcc, fps, size)
            if writer.isOpened():
                self._writer = writer
                self._path = path
                self._frames_written = 0
                logger.info("recording started: %s (%s)", path, fourcc_name)
                return path
            last_error = f"codec {fourcc_name} unavailable"
            writer.release()

        raise RuntimeError(f"could not start recording: {last_error}")

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            return
        self._writer.write(frame)
        self._frames_written += 1

    def stop(self) -> Path | None:
        """Finish the recording; returns the file path (or None if idle)."""
        if self._writer is None:
            return None
        self._writer.release()
        self._writer = None
        logger.info("recording stopped: %s (%d frames)", self._path, self._frames_written)
        path, self._path = self._path, None
        return path
