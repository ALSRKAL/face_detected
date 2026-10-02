"""Timestamped snapshots of annotated frames."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from facedetected.config import SnapshotConfig

logger = logging.getLogger(__name__)


class SnapshotManager:
    """Writes annotated frames into the snapshot directory.

    Filenames are ``{prefix}_{YYYYmmdd_HHMMSS}_{tag}.jpg``; a numeric suffix
    is appended when two snapshots share the same second.
    """

    def __init__(self, cfg: SnapshotConfig | None = None) -> None:
        self.cfg = cfg or SnapshotConfig()
        self.saved_count = 0

    def save(self, img: np.ndarray, tag: str = "snapshot") -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = f"{self.cfg.prefix}_{stamp}_{tag}"
        path = self.cfg.directory / f"{base}.jpg"

        counter = 1
        while path.exists():
            path = self.cfg.directory / f"{base}_{counter}.jpg"
            counter += 1

        if not cv2.imwrite(str(path), img):
            raise IOError(f"failed to write snapshot: {path}")
        self.saved_count += 1
        logger.info("snapshot saved: %s", path)
        return path
