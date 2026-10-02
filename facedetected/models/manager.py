"""Download, cache and integrity-check the MediaPipe model files.

The models are not committed to the repository (keeps clones light); they are
fetched once into a local cache directory and pinned by SHA-256 so a
corrupted or tampered download never reaches the inference engine.

The downloader is SSRF-hardened: only plain ``http``/``https`` URLs are
accepted, every resolved address is checked against loopback / private /
reserved / link-local ranges, and redirects to non-public hosts are refused.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
import socket
import ssl
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1 << 16
_DOWNLOAD_TIMEOUT_S = 60.0

_ALLOWED_SCHEMES = ("http", "https")


class ModelError(RuntimeError):
    """Base class for model management failures."""


class ModelDownloadError(ModelError):
    """The model file could not be fetched."""


class ModelIntegrityError(ModelError):
    """The downloaded file does not match the pinned SHA-256 digest."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A pinned model asset."""

    key: str
    url: str
    sha256: str
    filename: str
    description: str


FACE_LANDMARKER = ModelSpec(
    key="face_landmarker",
    url=(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    filename="face_landmarker.task",
    description="478-point face landmark model (MediaPipe FaceLandmarker)",
)

FACE_DETECTOR = ModelSpec(
    key="face_detector",
    url=(
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    ),
    sha256="b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
    filename="blaze_face_short_range.tflite",
    description="Short-range face detector (MediaPipe FaceDetector)",
)

MODELS: dict[str, ModelSpec] = {
    FACE_LANDMARKER.key: FACE_LANDMARKER,
    FACE_DETECTOR.key: FACE_DETECTOR,
}


def default_cache_dir() -> Path:
    """Model cache directory, overridable via ``FACEDETECTED_MODELS_DIR``."""
    env = os.environ.get("FACEDETECTED_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "facedetected" / "models"


def _assert_public_https_url(url: str) -> None:
    """Reject any URL whose scheme or resolved host is unsafe to fetch."""
    parts = urlsplit(url)
    if parts.scheme not in _ALLOWED_SCHEMES:
        raise ModelDownloadError(
            f"refusing to download from scheme {parts.scheme!r} (allowed: {_ALLOWED_SCHEMES})"
        )
    host = parts.hostname
    if not host:
        raise ModelDownloadError(f"URL has no hostname: {url!r}")
    if host.lower() in ("localhost", "localhost.localdomain") or host.endswith(".local"):
        raise ModelDownloadError(f"refusing to download from host {host!r}")

    try:
        addr_infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ModelDownloadError(f"cannot resolve host {host!r}: {exc}") from exc

    for info in addr_infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ModelDownloadError(
                f"host {host!r} resolves to a non-public address ({ip}); refusing download"
            )


def _download(spec: ModelSpec, destination: Path) -> Path:
    _assert_public_https_url(spec.url)
    logger.info("Downloading %s (%s)", spec.description, spec.url)
    request = Request(spec.url, headers={"User-Agent": "facedetected-model-manager"})
    context = ssl.create_default_context()
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".part", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urlopen(  # noqa: S310 - scheme and host validated above
                request, timeout=_DOWNLOAD_TIMEOUT_S, context=context
            ) as response, tmp:
                while True:
                    chunk = response.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    tmp.write(chunk)
            tmp_path.replace(destination)
        except (URLError, OSError) as exc:
            tmp_path.unlink(missing_ok=True)
            raise ModelDownloadError(f"failed to download {spec.url}: {exc}") from exc
    return destination


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(
    spec: ModelSpec, cache_dir: Path | None = None, *, force: bool = False
) -> Path:
    """Return a local path to the model, downloading it if needed.

    Raises:
        ModelIntegrityError: if an existing cached file fails verification
            and ``force`` is False, or if a fresh download fails verification.
    """
    cache = (cache_dir or default_cache_dir()).expanduser()
    target = cache / spec.filename

    if target.exists() and not force:
        actual = _sha256_of(target)
        if actual == spec.sha256:
            logger.debug("Model %s already cached at %s", spec.key, target)
            return target
        logger.warning(
            "Cached %s failed integrity check (sha256 mismatch); re-downloading", spec.key
        )

    _download(spec, target)
    actual = _sha256_of(target)
    if actual != spec.sha256:
        target.unlink(missing_ok=True)
        raise ModelIntegrityError(
            f"downloaded {spec.key} has sha256 {actual}, expected {spec.sha256}"
        )
    logger.info("Model %s ready at %s", spec.key, target)
    return target


def is_cached(spec: ModelSpec, cache_dir: Path | None = None) -> bool:
    """True when the model is present locally and passes verification."""
    cache = (cache_dir or default_cache_dir()).expanduser()
    target = cache / spec.filename
    return target.exists() and _sha256_of(target) == spec.sha256
