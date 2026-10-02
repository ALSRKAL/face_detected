"""Centralised logging configuration.

The old script spammed ``print`` for every one of the 468 landmarks on every
frame.  Everything now goes through :mod:`logging` so verbosity is a single
CLI flag (``-v`` for info, ``-vv`` for debug) and library users can plug the
toolkit into their own logging setup.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"

_RESET = "\033[0m"
_COLORS = {
    logging.DEBUG: "\033[38;5;245m",
    logging.INFO: "\033[38;5;39m",
    logging.WARNING: "\033[38;5;214m",
    logging.ERROR: "\033[38;5;203m",
    logging.CRITICAL: "\033[1;38;5;203m",
}


class _ColorFormatter(logging.Formatter):
    def __init__(self, fmt: str, datefmt: str, use_color: bool) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self._use_color:
            return text
        color = _COLORS.get(record.levelno, "")
        return f"{color}{text}{_RESET}"


def setup_logging(verbosity: int = 0, *, force: bool = False) -> None:
    """Configure the root logger.

    Args:
        verbosity: 0 -> WARNING, 1 -> INFO, 2+ -> DEBUG.
        force: remove handlers already attached to the root logger.
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    root = logging.getLogger()
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    if root.handlers:
        # Library embedded into another application: only adjust the level.
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        _ColorFormatter(_FORMAT, _DATEFMT, use_color=sys.stderr.isatty())
    )
    root.addHandler(handler)
    root.setLevel(level)

    # MediaPipe is chatty on DEBUG; keep its C++ logs quieter than ours.
    logging.getLogger("mediapipe").setLevel(
        max(level, logging.WARNING) if level == logging.DEBUG else logging.ERROR
    )
