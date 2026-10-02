"""Config dataclass validation."""

import pytest

from facedetected.config import (
    BlinkConfig,
    DetectorConfig,
    HudConfig,
    MouthConfig,
    OverlayConfig,
    SnapshotConfig,
    SourceConfig,
)


class TestDetectorConfig:
    def test_defaults_are_valid(self):
        cfg = DetectorConfig()
        assert cfg.num_faces == 1

    def test_num_faces_must_be_positive(self):
        with pytest.raises(ValueError):
            DetectorConfig(num_faces=0)

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            DetectorConfig(min_face_detection_confidence=1.5)
        with pytest.raises(ValueError):
            DetectorConfig(min_tracking_confidence=-0.1)


class TestSourceConfig:
    def test_defaults(self):
        cfg = SourceConfig()
        assert cfg.camera_index == 0
        assert cfg.max_reconnect_attempts == 0

    def test_negative_reconnect_rejected(self):
        with pytest.raises(ValueError):
            SourceConfig(reconnect_delay_s=-1)
        with pytest.raises(ValueError):
            SourceConfig(max_reconnect_attempts=-1)


class TestOverlayConfig:
    def test_defaults(self):
        cfg = OverlayConfig()
        assert cfg.draw_mesh and cfg.draw_boxes

    def test_thickness_bounds(self):
        with pytest.raises(ValueError):
            OverlayConfig(mesh_thickness=0)
        with pytest.raises(ValueError):
            OverlayConfig(box_thickness=-2)


class TestHudConfig:
    def test_opacity_bounds(self):
        with pytest.raises(ValueError):
            HudConfig(panel_opacity=1.2)
        with pytest.raises(ValueError):
            HudConfig(panel_opacity=-0.1)

    def test_fps_history(self):
        with pytest.raises(ValueError):
            HudConfig(fps_history=0)


class TestSnapshotConfig:
    def test_creates_directory(self, tmp_path):
        target = tmp_path / "sub" / "snaps"
        cfg = SnapshotConfig(directory=target)
        assert target.is_dir()
        assert cfg.prefix == "face"

    def test_empty_prefix_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            SnapshotConfig(directory=tmp_path, prefix="")


class TestBlinkConfig:
    def test_threshold_ordering(self):
        with pytest.raises(ValueError):
            BlinkConfig(ear_closed_threshold=0.5, ear_open_threshold=0.2)
        with pytest.raises(ValueError):
            BlinkConfig(ear_closed_threshold=0.0)

    def test_min_closed_frames(self):
        with pytest.raises(ValueError):
            BlinkConfig(min_closed_frames=0)


class TestMouthConfig:
    def test_threshold_bounds(self):
        with pytest.raises(ValueError):
            MouthConfig(mar_open_threshold=0.0)
        with pytest.raises(ValueError):
            MouthConfig(mar_open_threshold=5.0)
