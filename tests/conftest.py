"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FACE = PROJECT_ROOT / "tests" / "data" / "face_sample.jpg"
LEGACY_PHOTO = PROJECT_ROOT / "photo.jpg"


@pytest.fixture(scope="session")
def sample_face_path() -> Path:
    """Small public-domain portrait committed for deterministic integration tests."""
    assert SAMPLE_FACE.is_file(), f"missing test asset: {SAMPLE_FACE}"
    return SAMPLE_FACE


@pytest.fixture(scope="session")
def legacy_photo_path() -> Path:
    """The original repo screenshot (mesh-overlaid, kept for history)."""
    assert LEGACY_PHOTO.is_file(), f"missing legacy asset: {LEGACY_PHOTO}"
    return LEGACY_PHOTO
