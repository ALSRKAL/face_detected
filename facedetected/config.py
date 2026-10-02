"""Typed configuration for the detection pipeline.

Every knob of the toolkit lives in a small frozen dataclass so that the CLI,
the library API and the tests all share one source of truth.  Instances are
validated on construction: bad values fail fast instead of producing a
silently broken capture loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _check_confidence(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0.0, 1.0], got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Settings forwarded to the MediaPipe models."""

    num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    output_blendshapes: bool = False

    def __post_init__(self) -> None:
        if self.num_faces < 1:
            raise ValueError(f"num_faces must be >= 1, got {self.num_faces!r}")
        for name in (
            "min_face_detection_confidence",
            "min_face_presence_confidence",
            "min_tracking_confidence",
        ):
            _check_confidence(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Settings for the frame source (camera index, video file or image)."""

    camera_index: int = 0
    target_width: int | None = None
    target_height: int | None = None
    reconnect_delay_s: float = 2.0
    max_reconnect_attempts: int = 0  # 0 = retry forever

    def __post_init__(self) -> None:
        if self.reconnect_delay_s < 0:
            raise ValueError("reconnect_delay_s must be >= 0")
        if self.max_reconnect_attempts < 0:
            raise ValueError("max_reconnect_attempts must be >= 0")


@dataclass(frozen=True, slots=True)
class OverlayConfig:
    """What gets drawn on top of each frame."""

    draw_mesh: bool = True
    draw_tesselation: bool = False
    draw_iris: bool = False
    draw_boxes: bool = True
    show_landmark_ids: bool = False
    mesh_thickness: int = 1
    mesh_point_radius: int = 1
    box_thickness: int = 2
    mesh_color_bgr: tuple[int, int, int] = (200, 255, 255)
    box_color_bgr: tuple[int, int, int] = (80, 220, 60)
    text_color_bgr: tuple[int, int, int] = (255, 255, 255)
    accent_color_bgr: tuple[int, int, int] = (60, 200, 255)

    def __post_init__(self) -> None:
        for name in ("mesh_thickness", "mesh_point_radius", "box_thickness"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass(frozen=True, slots=True)
class HudConfig:
    """On-screen statistics panel."""

    enabled: bool = True
    show_fps: bool = True
    show_face_count: bool = True
    show_blink_count: bool = True
    show_mouth_state: bool = True
    show_head_pose: bool = True
    panel_opacity: float = 0.55
    fps_history: int = 30

    def __post_init__(self) -> None:
        if not 0.0 <= self.panel_opacity <= 1.0:
            raise ValueError("panel_opacity must be within [0.0, 1.0]")
        if self.fps_history < 1:
            raise ValueError("fps_history must be >= 1")


@dataclass(frozen=True, slots=True)
class SnapshotConfig:
    """Where snapshots and recordings land and how they are named."""

    directory: Path = field(default_factory=lambda: Path("outputs"))
    prefix: str = "face"
    include_timestamp: bool = True

    def __post_init__(self) -> None:
        if not self.prefix:
            raise ValueError("prefix must not be empty")
        self.directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class BlinkConfig:
    """Eye Aspect Ratio thresholds for the blink state machine."""

    ear_closed_threshold: float = 0.20
    ear_open_threshold: float = 0.25
    min_closed_frames: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.ear_closed_threshold < self.ear_open_threshold <= 1.0:
            raise ValueError(
                "require 0 < ear_closed_threshold < ear_open_threshold <= 1"
            )
        if self.min_closed_frames < 1:
            raise ValueError("min_closed_frames must be >= 1")


@dataclass(frozen=True, slots=True)
class MouthConfig:
    """Mouth Aspect Ratio threshold for open/closed classification."""

    mar_open_threshold: float = 0.55

    def __post_init__(self) -> None:
        if not 0.0 < self.mar_open_threshold <= 2.0:
            raise ValueError("mar_open_threshold must be within (0.0, 2.0]")
