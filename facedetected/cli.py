"""Command-line interface: ``facedetected run|image|download-models``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import astuple
from pathlib import Path

import cv2

from facedetected import __version__
from facedetected.config import (
    DetectorConfig,
    HudConfig,
    OverlayConfig,
    SnapshotConfig,
    SourceConfig,
)
from facedetected.engine import FaceDetectorEngine, FaceLandmarkerEngine
from facedetected.io_sources import CameraSource, ImageSource, VideoFileSource
from facedetected.logging_setup import setup_logging
from facedetected.models import manager as model_manager
from facedetected.overlays.draw import draw_result

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facedetected",
        description=(
            "Real-time face analysis: 478-point mesh, snapshots, recording. "
            "Built on MediaPipe Tasks + OpenCV."
        ),
    )
    parser.add_argument("--version", action="version", version=f"facedetected {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="analyse a camera or a video file in real time")
    run_p.add_argument(
        "source", nargs="?", default="0",
        help="camera index (default: 0) or path to a video file",
    )
    run_p.add_argument("--faces", type=int, default=1, help="max faces to track (default: 1)")
    run_p.add_argument("--width", type=int, default=None, help="requested capture width")
    run_p.add_argument("--height", type=int, default=None, help="requested capture height")
    run_p.add_argument("--no-mesh", action="store_true", help="hide the face contour mesh")
    run_p.add_argument("--tesselation", action="store_true", help="draw the full 2556-edge mesh")
    run_p.add_argument("--no-boxes", action="store_true", help="hide bounding boxes")
    run_p.add_argument("--iris", action="store_true", help="highlight the iris rings")
    run_p.add_argument("--ids", action="store_true", help="draw numeric landmark ids (debug)")
    run_p.add_argument("--no-hud", action="store_true", help="hide the stats panel")
    run_p.add_argument("--record", action="store_true", help="start recording immediately")
    run_p.add_argument("--loop", action="store_true", help="loop video files forever")
    run_p.add_argument("--out-dir", default="outputs", help="snapshot/recording directory")
    run_p.add_argument(
        "--headless", action="store_true",
        help="no preview window; process the whole stream then exit (servers/CI)",
    )
    run_p.add_argument(
        "--max-frames", type=int, default=None,
        help="stop after this many frames (0 = unlimited)",
    )
    _add_verbosity(run_p)

    img_p = sub.add_parser("image", help="analyse still image(s) and save annotated copies")
    img_p.add_argument("paths", nargs="+", help="image file(s) to process")
    img_p.add_argument("--faces", type=int, default=5, help="max faces per image (default: 5)")
    img_p.add_argument("--out-dir", default="outputs", help="where annotated images go")
    img_p.add_argument("--no-save", action="store_true", help="do not write annotated files")
    img_p.add_argument("--json", action="store_true", help="print a JSON summary to stdout")
    img_p.add_argument("--detector", action="store_true", help="boxes only (faster, no mesh)")
    _add_verbosity(img_p)

    dl_p = sub.add_parser("download-models", help="pre-fetch the MediaPipe model files")
    dl_p.add_argument("--models-dir", default=None, help="cache directory (default: ~/.cache)")
    _add_verbosity(dl_p)

    return parser


def _add_verbosity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug")


# -- subcommand handlers ------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    source_cfg = SourceConfig(
        camera_index=0, target_width=args.width, target_height=args.height
    )
    detector_cfg = DetectorConfig(num_faces=args.faces)
    overlay_cfg = OverlayConfig(
        draw_mesh=not args.no_mesh,
        draw_tesselation=args.tesselation,
        draw_boxes=not args.no_boxes,
        draw_iris=args.iris,
        show_landmark_ids=args.ids,
    )
    hud_cfg = HudConfig(enabled=not args.no_hud)
    snapshot_cfg = SnapshotConfig(directory=Path(args.out_dir))

    if _is_camera(args.source):
        source = CameraSource(source_cfg, index=int(args.source))
    else:
        source = VideoFileSource(args.source, loop=args.loop)

    engine = FaceLandmarkerEngine(detector_cfg, mode="video")
    from facedetected.app import FaceAnalysisApp

    app = FaceAnalysisApp(
        source,
        engine,
        overlay_cfg=overlay_cfg,
        hud_cfg=hud_cfg,
        snapshot_cfg=snapshot_cfg,
        headless=args.headless,
        max_frames=args.max_frames or None,
        auto_record=args.record,
    )
    summary = app.run()
    print(summary.as_text())
    return 0


def _cmd_image(args: argparse.Namespace) -> int:
    detector_cfg = DetectorConfig(num_faces=args.faces)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for raw_path in args.paths:
        path = Path(raw_path)
        try:
            with ImageSource(path) as src:
                frame = src.read()
        except (FileNotFoundError, RuntimeError) as exc:
            logger.error("%s: %s", path, exc)
            results.append({"path": str(path), "error": str(exc)})
            continue

        if args.detector:
            with FaceDetectorEngine(detector_cfg, mode="image") as engine:
                result = engine.detect_image(frame.image)
        else:
            with FaceLandmarkerEngine(detector_cfg, mode="image") as engine:
                result = engine.detect_image(frame.image)

        draw_result(frame.image, result, OverlayConfig(draw_mesh=not args.detector))
        annotated_path = None
        if not args.no_save:
            annotated_path = out_dir / f"{path.stem}_annotated{path.suffix or '.jpg'}"
            if not cv2.imwrite(str(annotated_path), frame.image):
                logger.error("failed to write %s", annotated_path)
                annotated_path = None

        entry = {
            "path": str(path),
            "faces": result.face_count,
            "landmarks": [len(f.landmarks) for f in result.faces],
            "boxes": [list(astuple(f.box)) for f in result.faces],
            "annotated": str(annotated_path) if annotated_path else None,
        }
        results.append(entry)
        logger.info(
            "%s: %d face(s)%s", path, result.face_count,
            f" -> {annotated_path}" if annotated_path else "",
        )

    if args.json:
        print(json.dumps(results, indent=2))
    return 0 if all("error" not in r for r in results) else 1


def _cmd_download_models(args: argparse.Namespace) -> int:
    from facedetected.models.manager import FACE_DETECTOR, FACE_LANDMARKER, ensure_model

    cache = Path(args.models_dir).expanduser() if args.models_dir else None
    for spec in (FACE_LANDMARKER, FACE_DETECTOR):
        path = ensure_model(spec, cache)
        print(f"{spec.key}: {path}")
    return 0


def _is_camera(source: str) -> bool:
    try:
        int(source)
        return True
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose, force=True)

    handlers = {
        "run": _cmd_run,
        "image": _cmd_image,
        "download-models": _cmd_download_models,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        logger.info("interrupted")
        return 130
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
