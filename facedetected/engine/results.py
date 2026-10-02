"""Typed detection results shared by every engine and overlay."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Landmark:
    """One face-mesh point in both pixel and normalised coordinates."""

    x: float  # pixels
    y: float
    z: float  # pixels, relative scale (depth proxy)
    nx: float  # normalised [0..1] across image width
    ny: float
    nz: float

    @property
    def ix(self) -> int:
        return int(round(self.x))

    @property
    def iy(self) -> int:
        return int(round(self.y))


@dataclass(frozen=True, slots=True)
class FaceBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def top_left(self) -> tuple[int, int]:
        return self.x, self.y

    @property
    def area(self) -> int:
        return self.w * self.h

    def clipped_to(self, width: int, height: int) -> "FaceBox":
        x0 = max(0, min(self.x, width - 1))
        y0 = max(0, min(self.y, height - 1))
        x1 = max(x0 + 1, min(self.x2, width))
        y1 = max(y0 + 1, min(self.y2, height))
        return FaceBox(x0, y0, x1 - x0, y1 - y0)


@dataclass(frozen=True, slots=True)
class Face:
    """A single detected face with its landmarks (and optional extras)."""

    index: int
    box: FaceBox
    landmarks: tuple[Landmark, ...]
    score: float | None = None
    blendshapes: dict[str, float] | None = None

    @property
    def has_landmarks(self) -> bool:
        return len(self.landmarks) > 0

    def landmark(self, index: int) -> Landmark:
        return self.landmarks[index]


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Everything one frame produced."""

    faces: tuple[Face, ...]
    width: int
    height: int
    timestamp_ms: int

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def primary(self) -> Face | None:
        """Largest face in the frame (conventionally the closest person)."""
        if not self.faces:
            return None
        return max(self.faces, key=lambda f: f.box.area)
