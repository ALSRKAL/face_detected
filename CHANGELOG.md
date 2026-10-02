# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0] - 2026-10-03

Complete rewrite of the original single-file script into a professional,
tested, installable Python package.

### Added
- **Package structure** (`facedetected/`) with typed frozen dataclass configs,
  installable via `pyproject.toml` and a `facedetected` console entry point.
- **Engines** on the MediaPipe Tasks API (the legacy `mp.solutions` API is
  removed in MediaPipe 1.0):
  - `FaceLandmarkerEngine` — 478-point dense mesh, image/video/live-stream modes.
  - `FaceDetectorEngine` — fast BlazeFace bounding boxes with confidence.
- **Face analytics**: blink counting via Eye Aspect Ratio with a hysteretic
  state machine, mouth open/closed via Mouth Aspect Ratio, and head pose
  (yaw/pitch/roll) via 6-point `solvePnP` in image-space axes.
- **Frame sources**: `CameraSource` (auto-reconnect, configurable retries),
  `VideoFileSource` (loop option), `ImageSource` — behind one `FrameSource` API.
- **Interactive app**: keyboard controls (quit/snapshot/record/mesh/boxes/HUD/ids),
  translucent HUD panel with smoothed FPS, session summary statistics.
- **Snapshots** (timestamped, collision-safe) and **MP4 recording** with
  codec negotiation (`mp4v` → `avc1` → `XVID` → `MJPG`).
- **Web preview** (`--web [HOST:PORT]`): MJPEG streaming of the annotated
  frames to any browser, loopback-bound by default — ideal for headless
  servers and Wayland setups without a working highgui window.
- **CLI**: `facedetected run | image | download-models` plus
  `python -m facedetected`; JSON summaries for image batches; headless mode.
- **Model manager**: models fetched on demand into a cache dir, pinned by
  SHA-256, with an SSRF-hardened downloader (scheme/host/address validation).
- **Tests**: 68 pytest tests (unit + integration) and a GitHub Actions CI
  workflow (ruff lint + test matrix on Python 3.10–3.12) with dependabot.
- **Docs**: professional README (EN + AR), examples, MIT LICENSE.

### Changed
- FPS is measured with an exponentially smoothed meter instead of the jumpy
  `1 / (now - last)` delta.
- Per-frame `print` of every landmark replaced by structured `logging`
  with verbosity flags.

### Removed
- Tracked `.idea/` IDE configuration (kept locally, now gitignored).

### Kept
- The original `photo.jpg` demo screenshot, unchanged, for history.
