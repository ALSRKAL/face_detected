"""Per-face analytics aggregator wired into the app loop."""

from __future__ import annotations

from dataclasses import dataclass

from facedetected.analytics.blink import BlinkDetector, mean_ear
from facedetected.analytics.head_pose import HeadPoseEstimator
from facedetected.analytics.mouth import MouthDetector, mouth_aspect_ratio
from facedetected.config import BlinkConfig, MouthConfig
from facedetected.engine.results import DetectionResult


@dataclass(frozen=True, slots=True)
class FaceStats:
    """Everything the analytics layer knows about one face, this frame."""

    ear: float | None
    mar: float | None
    eye_closed: bool
    blink_count: int
    mouth_open: bool
    pose: tuple[float, float, float] | None


class _FaceTracker:
    """Bundle of per-face state machines."""

    def __init__(self, blink_cfg: BlinkConfig, mouth_cfg: MouthConfig) -> None:
        self.blink = BlinkDetector(blink_cfg)
        self.mouth = MouthDetector(mouth_cfg)
        self.pose = HeadPoseEstimator()
        self.last: FaceStats | None = None


class FaceAnalytics:
    """Updates trackers for every visible face and keeps cumulative counts.

    Blink totals survive a face briefly leaving the frame; the trackers are
    recreated (stateless parts) while the cumulative counter is kept by index.
    """

    def __init__(
        self,
        blink_cfg: BlinkConfig | None = None,
        mouth_cfg: MouthConfig | None = None,
    ) -> None:
        self.blink_cfg = blink_cfg or BlinkConfig()
        self.mouth_cfg = mouth_cfg or MouthConfig()
        self._trackers: dict[int, _FaceTracker] = {}
        self._blink_totals: dict[int, int] = {}
        self.last_blinks_delta = 0

    def update(self, result: DetectionResult) -> dict[int, FaceStats]:
        """Feed one frame; returns per-face stats keyed by face index."""
        stats: dict[int, FaceStats] = {}
        present = set()
        blinks_this_frame = 0

        for face in result.faces:
            present.add(face.index)
            tracker = self._trackers.get(face.index)
            if tracker is None:
                tracker = _FaceTracker(self.blink_cfg, self.mouth_cfg)
                self._trackers[face.index] = tracker

            ear = mean_ear(face.landmarks, (33, 160, 158, 133, 153, 144), (362, 385, 387, 263, 373, 380))
            mar = mouth_aspect_ratio(face.landmarks)
            eye_closed, blinked = tracker.blink.update(ear)
            mouth_open = tracker.mouth.update(mar)
            pose = tracker.pose.update(face, result.width, result.height)
            if blinked:
                self._blink_totals[face.index] = self._blink_totals.get(face.index, 0) + 1
                blinks_this_frame += 1

            tracker.last = FaceStats(
                ear=ear,
                mar=mar,
                eye_closed=eye_closed,
                blink_count=self._blink_totals.get(face.index, 0),
                mouth_open=mouth_open,
                pose=pose,
            )
            stats[face.index] = tracker.last

        # Faces that left the frame keep their blink total but go stale.
        for index in list(self._trackers):
            if index not in present:
                self._trackers[index].last = None
        self.last_blinks_delta = blinks_this_frame
        return stats

    def stats_for(self, face_index: int) -> FaceStats | None:
        tracker = self._trackers.get(face_index)
        return tracker.last if tracker else None

    def total_blinks(self, face_index: int | None = None) -> int:
        """Cumulative blinks for one face, or across all faces when None."""
        if face_index is None:
            return sum(self._blink_totals.values())
        return self._blink_totals.get(face_index, 0)

    def primary_stats(self, result: DetectionResult) -> FaceStats | None:
        """Stats of the primary (largest) face from the last update call."""
        primary = result.primary
        if primary is None:
            return None
        return self.stats_for(primary.index)
