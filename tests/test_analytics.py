"""Analytics: EAR/MAR math, blink & mouth state machines, head pose."""

from types import SimpleNamespace

import pytest

from facedetected.analytics.blink import BlinkDetector, eye_aspect_ratio, mean_ear
from facedetected.analytics.face_analytics import FaceAnalytics
from facedetected.analytics.head_pose import HeadPoseEstimator, format_pose
from facedetected.analytics.landmarks import POSE_SEXTET
from facedetected.analytics.mouth import MouthDetector, mouth_aspect_ratio
from facedetected.config import BlinkConfig
from facedetected.engine.results import DetectionResult, Face, FaceBox, Landmark

LEFT = (33, 160, 158, 133, 153, 144)
RIGHT = (362, 385, 387, 263, 373, 380)


def lm(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=0.0)


def face_with_eyes(n_points: int = 400, *, vertical: float = 10.0) -> list:
    """Landmark list with an 'open' (vertical=10) or 'closed' (vertical=1) eye pair."""
    v = vertical
    lms = [lm(0, 0)] * n_points
    for i, (x, y) in {
        33: (0, 0), 133: (40, 0), 160: (20, -v), 158: (20, -v),
        153: (20, v), 144: (20, v),
        362: (100, 0), 263: (140, 0), 385: (120, -v), 387: (120, -v),
        373: (120, v), 380: (120, v),
    }.items():
        lms[i] = lm(x, y)
    return lms


class TestEar:
    def test_open_eye(self):
        ear = eye_aspect_ratio(face_with_eyes(vertical=10.0), LEFT)
        assert ear == pytest.approx(0.5)

    def test_closed_eye(self):
        ear = eye_aspect_ratio(face_with_eyes(vertical=1.0), LEFT)
        assert ear == pytest.approx(0.05)

    def test_mean_over_both_eyes(self):
        lms = face_with_eyes(vertical=8.0)
        assert mean_ear(lms, LEFT, RIGHT) == pytest.approx(0.4)

    def test_degenerate_returns_none(self):
        lms = [lm(0, 0)] * 400  # all coincident
        assert eye_aspect_ratio(lms, LEFT) is None
        assert mean_ear(lms, LEFT, RIGHT) is None


class TestBlinkDetector:
    def test_single_blink(self):
        bd = BlinkDetector()
        closed = eye_aspect_ratio(face_with_eyes(vertical=1.0), LEFT)
        opened = eye_aspect_ratio(face_with_eyes(vertical=10.0), LEFT)
        for _ in range(3):
            is_closed, _ = bd.update(closed)
        assert is_closed
        _, blinked = bd.update(opened)
        assert blinked and bd.blink_count == 1

    def test_noise_does_not_chatter(self):
        bd = BlinkDetector()
        closed_low = eye_aspect_ratio(face_with_eyes(vertical=1.0), LEFT)
        closed_noisy = closed_low + 0.03  # still below closed threshold (0.2)
        opened = eye_aspect_ratio(face_with_eyes(vertical=10.0), LEFT)
        for ear in (closed_low, closed_noisy, closed_low, closed_noisy, closed_low):
            bd.update(ear)
        bd.update(opened)
        assert bd.blink_count == 1

    def test_none_measurement_holds_state(self):
        bd = BlinkDetector()
        closed = eye_aspect_ratio(face_with_eyes(vertical=1.0), LEFT)
        bd.update(closed)
        bd.update(closed)
        bd.update(None)
        assert bd.is_closed
        assert bd.blink_count == 0

    def test_strict_thresholds_rejected(self):
        with pytest.raises(ValueError):
            BlinkConfig(ear_closed_threshold=0.3, ear_open_threshold=0.3)


class TestMouth:
    def test_mar_values(self):
        lms = [lm(0, 0)] * 400
        for i, (x, y) in {61: (0, 0), 291: (50, 0), 13: (25, -2), 14: (25, 2)}.items():
            lms[i] = lm(x, y)
        assert mouth_aspect_ratio(lms) == pytest.approx(0.08)

    def test_hysteresis(self):
        md = MouthDetector()
        closed, open_ = 0.08, 0.60
        assert md.update(closed) is False
        assert md.update(open_) is True
        # between close_threshold (0.44) and open threshold: stays open
        assert md.update(0.5) is True
        assert md.update(closed) is False

    def test_none_holds_state(self):
        md = MouthDetector()
        md.update(0.9)
        assert md.update(None) is True


def synthetic_face(index: int = 0) -> Face:
    """Face whose six pose landmarks are the exact projection of the 3D
    model with identity rotation, so PnP must recover ~zero angles."""
    import numpy as np

    from facedetected.analytics.head_pose import _MODEL_POINTS

    fx = fy = 480.0
    cx, cy = 240.0, 320.0
    camera = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    # model pushed 1000 units in front of the camera, no rotation
    world = _MODEL_POINTS + np.array([0.0, 0.0, 1000.0])
    projected = (camera @ world.T).T
    pixels = projected[:, :2] / projected[:, 2:3]

    lms = [Landmark(x=0, y=0, z=0, nx=0, ny=0, nz=0)] * 478
    for sextet_index, (u, v) in zip(POSE_SEXTET, pixels, strict=True):
        lms[sextet_index] = Landmark(x=float(u), y=float(v), z=0,
                                     nx=u / 480, ny=v / 640, nz=0)
    return Face(index=index, box=FaceBox(120, 150, 240, 280), landmarks=tuple(lms))


class TestHeadPose:
    def test_frontal_face_near_zero(self):
        pose = HeadPoseEstimator().update(synthetic_face(), 480, 640)
        assert pose is not None
        yaw, pitch, roll = pose
        assert abs(yaw) < 5
        assert abs(pitch) < 5
        assert abs(roll) < 5

    def test_format_pose(self):
        assert format_pose(None) is None
        text = format_pose((1.2, -2.3, 3.4))
        assert text.startswith("y+")

    def test_empty_face_returns_none(self):
        empty = Face(index=0, box=FaceBox(0, 0, 1, 1), landmarks=())
        assert HeadPoseEstimator().update(empty, 480, 640) is None


class TestFaceAnalytics:
    def test_cumulative_blinks_survive_gap(self):
        analytics = FaceAnalytics()

        def frame(lms):
            f = Face(index=0, box=FaceBox(0, 0, 100, 100), landmarks=tuple(lms))
            return DetectionResult(faces=(f,), width=640, height=480, timestamp_ms=0)

        # two closed frames, then an open one -> 1 blink
        analytics.update(frame(face_with_eyes(vertical=1.0)))
        analytics.update(frame(face_with_eyes(vertical=1.0)))
        stats = analytics.update(frame(face_with_eyes(vertical=10.0)))[0]
        assert stats.blink_count == 1
        assert analytics.total_blinks(0) == 1

        # face leaves for a frame, returns -> total must persist
        analytics.update(DetectionResult(faces=(), width=640, height=480, timestamp_ms=1))
        assert analytics.stats_for(0) is None
        stats = analytics.update(frame(face_with_eyes(vertical=10.0)))[0]
        assert stats.blink_count == 1

    def test_delta_counter(self):
        analytics = FaceAnalytics()
        analytics.update(DetectionResult(faces=(), width=640, height=480, timestamp_ms=0))
        assert analytics.last_blinks_delta == 0

    def test_primary_stats(self):
        analytics = FaceAnalytics()
        empty = DetectionResult(faces=(), width=640, height=480, timestamp_ms=0)
        assert analytics.primary_stats(empty) is None
