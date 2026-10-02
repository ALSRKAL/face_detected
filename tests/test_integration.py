"""Integration tests: real models against the committed sample portrait.

These download the MediaPipe models on first run (network required) and are
skipped by default locally (``pytest``); CI runs them via ``pytest -m ''``.
"""

from __future__ import annotations

import json

import cv2
import pytest

from facedetected.analytics import FaceAnalytics
from facedetected.config import DetectorConfig
from facedetected.engine import FaceDetectorEngine, FaceLandmarkerEngine
from facedetected.io_sources import ImageSource

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def landmarker_result(sample_face_path):
    cfg = DetectorConfig(num_faces=2)
    with ImageSource(sample_face_path) as src:
        frame = src.read()
    with FaceLandmarkerEngine(cfg, mode="image") as engine:
        return frame, engine.detect_image(frame.image)


class TestLandmarkerEngine:
    def test_detects_single_face(self, landmarker_result):
        _, result = landmarker_result
        assert result.face_count == 1

    def test_full_mesh_present(self, landmarker_result):
        _, result = landmarker_result
        face = result.faces[0]
        assert len(face.landmarks) == 478

    def test_box_is_sane(self, landmarker_result):
        _, result = landmarker_result
        h, w = result.height, result.width
        box = result.faces[0].box
        assert 0 <= box.x < box.x2 <= w
        assert 0 <= box.y < box.y2 <= h
        assert box.area > 0.05 * w * h  # portrait face is prominent

    def test_eyes_open_ear(self, landmarker_result):
        from facedetected.analytics.blink import mean_ear

        _, result = landmarker_result
        lms = result.faces[0].landmarks
        ear = mean_ear(lms, (33, 160, 158, 133, 153, 144), (362, 385, 387, 263, 373, 380))
        assert ear is not None and ear > 0.12

    def test_head_pose_sensible(self, landmarker_result):
        _, result = landmarker_result
        stats = FaceAnalytics().update(result)
        pose = stats[0].pose
        assert pose is not None
        yaw, pitch, roll = pose
        assert abs(yaw) < 30 and abs(pitch) < 45 and abs(roll) < 30


class TestDetectorEngine:
    def test_detects_with_confidence(self, sample_face_path):
        with ImageSource(sample_face_path) as src:
            frame = src.read()
        with FaceDetectorEngine(mode="image") as engine:
            result = engine.detect_image(frame.image)
        assert result.face_count == 1
        assert result.faces[0].score is not None
        assert result.faces[0].score > 0.5


class TestLegacyPhoto:
    def test_legacy_photo_loads_but_has_no_clean_face(self, legacy_photo_path):
        """The historical README screenshot is already mesh-overlaid."""
        with ImageSource(legacy_photo_path) as src:
            frame = src.read()
        assert frame.image is not None
        with FaceLandmarkerEngine(DetectorConfig(num_faces=2), mode="image") as engine:
            result = engine.detect_image(frame.image)
        assert result.face_count == 0  # the overlay hides the real face


class TestVideoMode:
    def test_consecutive_frames_track(self, sample_face_path):
        img = cv2.imread(str(sample_face_path))
        with FaceLandmarkerEngine(DetectorConfig(num_faces=1), mode="video") as engine:
            first = engine.detect_video(img, timestamp_ms=0)
            second = engine.detect_video(img, timestamp_ms=33)
        assert first.face_count == 1
        assert second.face_count == 1
        assert second.timestamp_ms == 33


class TestCliImage:
    def test_image_json_output(self, sample_face_path, tmp_path, capsys):
        from facedetected.cli import main

        out_dir = tmp_path / "out"
        code = main(
            ["image", str(sample_face_path), "--out-dir", str(out_dir), "--json"]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload[0]["faces"] == 1
        assert payload[0]["landmarks"] == [478]
        assert (out_dir / "face_sample_annotated.jpg").is_file()

    def test_image_detector_mode(self, sample_face_path, capsys):
        from facedetected.cli import main

        code = main(
            ["image", str(sample_face_path), "--detector", "--json", "--no-save"]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload[0]["faces"] == 1
        assert payload[0]["annotated"] is None

    def test_missing_image_is_reported(self, capsys):
        from facedetected.cli import main

        code = main(["image", "/nonexistent/file.jpg", "--no-save", "--json"])
        assert code == 1
