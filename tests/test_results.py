"""Detection result helpers (boxes, primary face selection)."""

from facedetected.engine.results import DetectionResult, Face, FaceBox, Landmark


def make_face(index: int, box: FaceBox) -> Face:
    return Face(index=index, box=box, landmarks=())


class TestFaceBox:
    def test_properties(self):
        box = FaceBox(x=10, y=20, w=30, h=40)
        assert box.x2 == 40
        assert box.y2 == 60
        assert box.center == (25, 40)
        assert box.area == 1200

    def test_clip_inside(self):
        box = FaceBox(x=5, y=5, w=10, h=10)
        assert box.clipped_to(100, 100) == box

    def test_clip_overflow(self):
        box = FaceBox(x=-5, y=-5, w=200, h=200)
        clipped = box.clipped_to(100, 100)
        assert clipped.x == 0 and clipped.y == 0
        assert clipped.x2 <= 100 and clipped.y2 <= 100
        assert clipped.w > 0 and clipped.h > 0


class TestDetectionResult:
    def test_primary_is_largest(self):
        small = make_face(0, FaceBox(0, 0, 10, 10))
        big = make_face(1, FaceBox(0, 0, 100, 100))
        tiny = make_face(2, FaceBox(0, 0, 5, 5))
        result = DetectionResult(faces=(small, big, tiny), width=640, height=480, timestamp_ms=0)
        assert result.primary is big
        assert result.face_count == 3

    def test_primary_empty(self):
        result = DetectionResult(faces=(), width=640, height=480, timestamp_ms=0)
        assert result.primary is None
        assert result.face_count == 0


class TestLandmark:
    def test_pixel_rounding(self):
        lm = Landmark(x=10.6, y=20.4, z=0.0, nx=0.1, ny=0.2, nz=0.0)
        assert lm.ix == 11
        assert lm.iy == 20
