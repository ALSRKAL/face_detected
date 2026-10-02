"""Abstract frame source shared by camera, video-file and image readers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    """One frame plus its position in the stream."""

    image: np.ndarray  # BGR
    index: int
    timestamp_ms: int


class FrameSource(ABC):
    """Common lifecycle for all frame sources (context-manager friendly)."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._open = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_open(self) -> bool:
        return self._open

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @abstractmethod
    def open(self) -> None:
        """Acquire the underlying resource."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Return the next frame, or ``None`` when the stream is exhausted."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying resource."""
