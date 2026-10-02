"""CLI surface tests (parser only; no model downloads)."""

import pytest

from facedetected import __version__
from facedetected.cli import build_parser


class TestParser:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args(["--version"])
        assert exc.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_command_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_run_defaults(self):
        args = build_parser().parse_args(["run"])
        assert args.source == "0"
        assert args.faces == 1
        assert not args.headless
        assert not args.record
        assert not args.no_analytics  # analytics on by default

    def test_run_source_detection_inputs(self):
        args = build_parser().parse_args(["run", "2", "--faces", "3", "--headless", "--max-frames", "10"])
        assert args.source == "2"
        assert args.faces == 3
        assert args.headless
        assert args.max_frames == 10

    def test_image_options(self):
        args = build_parser().parse_args(["image", "a.jpg", "b.png", "--json", "--detector"])
        assert args.paths == ["a.jpg", "b.png"]
        assert args.json and args.detector

    def test_download_models(self):
        args = build_parser().parse_args(["download-models"])
        assert args.command == "download-models"
