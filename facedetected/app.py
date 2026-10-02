"""High-level application wiring source -> engine -> overlays -> output."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2

from facedetected.analytics.face_analytics import FaceAnalytics
from facedetected.analytics.head_pose import format_pose
from facedetected.config import HudConfig, OverlayConfig, SnapshotConfig
from facedetected.engine import FaceLandmarkerEngine
from facedetected.io_sources.base import Frame, FrameSource
from facedetected.overlays.draw import draw_result
from facedetected.overlays.hud import Hud
from facedetected.recording import Recorder
from facedetected.snapshot import SnapshotManager
from facedetected.utils.fps import FPSMeter

logger = logging.getLogger(__name__)

WINDOW_TITLE = "facedetected  |  q quit  s snapshot  r record  m mesh  b boxes  h hud  i ids"


@dataclass(frozen=True, slots=True)
class AppSummary:
    """What one session produced."""

    frames_processed: int
    elapsed_s: float
    avg_fps: float
    frames_with_faces: int
    total_face_detections: int
    total_blinks: int
    snapshots_saved: int
    recording_path: str | None

    def as_text(self) -> str:
        return (
            f"session summary: {self.frames_processed} frames in {self.elapsed_s:.1f}s "
            f"({self.avg_fps:.1f} FPS avg) | frames with faces: {self.frames_with_faces} "
            f"| face detections: {self.total_face_detections} "
            f"| blinks: {self.total_blinks} "
            f"| snapshots: {self.snapshots_saved}"
            + (f" | recording: {self.recording_path}" if self.recording_path else "")
        )


class FaceAnalysisApp:
    """Runs the interactive analysis loop over a frame source.

    Keyboard controls (when a window is available):
        q / ESC quit | s snapshot | r toggle recording
        m toggle mesh | b toggle boxes | h toggle HUD | i toggle landmark ids
    """

    def __init__(
        self,
        source: FrameSource,
        engine: FaceLandmarkerEngine,
        *,
        overlay_cfg: OverlayConfig | None = None,
        hud_cfg: HudConfig | None = None,
        snapshot_cfg: SnapshotConfig | None = None,
        headless: bool = False,
        max_frames: int | None = None,
        auto_record: bool = False,
        enable_analytics: bool = True,
    ) -> None:
        self.source = source
        self.engine = engine
        self.overlay_cfg = overlay_cfg or OverlayConfig()
        self.hud = Hud(hud_cfg or HudConfig())
        self.snapshots = SnapshotManager(snapshot_cfg or SnapshotConfig())
        self.recorder = Recorder(snapshot_cfg or SnapshotConfig())
        self.analytics = FaceAnalytics() if enable_analytics else None
        self.headless = headless
        self.max_frames = max_frames
        self.auto_record = auto_record
        self._source_fps = 30.0

    # -- public API ----------------------------------------------------------

    def run(self) -> AppSummary:
        fps_meter = FPSMeter()
        frames = 0
        frames_with_faces = 0
        total_faces = 0
        total_blinks = 0
        started = time.monotonic()

        with self.source, self.engine:
            self._source_fps = getattr(self.source, "fps", 30.0) or 30.0
            while self.max_frames is None or frames < self.max_frames:
                frame = self.source.read()
                if frame is None:
                    logger.info("source %s exhausted", self.source.name)
                    break
                summary_faces, frame_blinks = self._process_frame(frame, fps_meter)
                frames += 1
                total_faces += summary_faces
                total_blinks += frame_blinks
                if summary_faces:
                    frames_with_faces += 1
                if not self.headless:
                    key = cv2.waitKey(1) & 0xFF
                    if not self._handle_key(key, frame):
                        break

        elapsed = max(time.monotonic() - started, 1e-6)
        recording_path = self.recorder.stop()  # no-op when never started
        if not self.headless:
            cv2.destroyAllWindows()

        summary = AppSummary(
            frames_processed=frames,
            elapsed_s=elapsed,
            avg_fps=frames / elapsed,
            frames_with_faces=frames_with_faces,
            total_face_detections=total_faces,
            total_blinks=total_blinks,
            snapshots_saved=self.snapshots.saved_count,
            recording_path=str(recording_path) if recording_path else None,
        )
        logger.info(summary.as_text())
        return summary

    # -- internals -----------------------------------------------------------

    def _process_frame(self, frame: Frame, fps_meter: FPSMeter) -> tuple[int, int]:
        result = self.engine.detect_video(frame.image, frame.timestamp_ms)
        draw_result(frame.image, result, self.overlay_cfg)

        blinks_this_frame = 0
        blink_count = None
        mouth_open = None
        head_pose = None

        if self.analytics is not None:
            self.analytics.update(result)
            stats = self.analytics.primary_stats(result)
            if stats is not None:
                blink_count = stats.blink_count
                mouth_open = stats.mouth_open
                head_pose = format_pose(stats.pose)
            blinks_this_frame = self.analytics.last_blinks_delta

        if self.auto_record and not self.recorder.is_recording:
            h, w = frame.image.shape[:2]
            self.recorder.start(self._source_fps, (w, h))

        fps = fps_meter.tick()
        self.hud.update(
            fps=fps,
            face_count=result.face_count,
            blink_count=blink_count,
            mouth_open=mouth_open,
            head_pose=head_pose,
        )
        self.hud.draw(frame.image, recording=self.recorder.is_recording)
        self.hud.draw_help_footer(frame.image)

        if self.recorder.is_recording:
            self.recorder.write(frame.image)
        return result.face_count, blinks_this_frame

    def _handle_key(self, key: int, frame: Frame) -> bool:
        """Returns False when the loop should stop."""
        if key in (ord("q"), 27):  # ESC
            logger.info("quit requested")
            return False
        if key == ord("s"):
            self.snapshots.save(frame.image)
        elif key == ord("r"):
            if self.recorder.is_recording:
                self.recorder.stop()
            else:
                h, w = frame.image.shape[:2]
                self.recorder.start(self._source_fps, (w, h))
        elif key == ord("m"):
            self.overlay_cfg = _toggle(self.overlay_cfg, "draw_mesh")
        elif key == ord("b"):
            self.overlay_cfg = _toggle(self.overlay_cfg, "draw_boxes")
        elif key == ord("i"):
            self.overlay_cfg = _toggle(self.overlay_cfg, "show_landmark_ids")
        elif key == ord("h"):
            self.hud.toggle()
        return True

    def snapshot_current(self, frame: Frame) -> None:
        self.snapshots.save(frame.image)


def _toggle(cfg: OverlayConfig, field: str) -> OverlayConfig:
    """OverlayConfig is frozen; produce a copy with one boolean flipped."""
    from dataclasses import replace

    return replace(cfg, **{field: not getattr(cfg, field)})
