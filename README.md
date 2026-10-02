# facedetected

[![CI](https://github.com/ALSRKAL/face_detected/actions/workflows/ci.yml/badge.svg)](https://github.com/ALSRKAL/face_detected/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://docs.astral.sh/ruff/)

**Real-time face analysis toolkit** — 478-point dense face mesh, blink & mouth detection,
head pose, snapshots and recording. Built on the modern **MediaPipe Tasks API** and **OpenCV**.

> v2.0 is a complete rewrite of the original single-file experiment
> ([the old `facedetected.py`](https://github.com/ALSRKAL/face_detected/commit/6719845))
> into a typed, tested, installable Python package.

---

## Features

| Area | What you get |
|---|---|
| **Detection** | Dense 478-landmark face mesh + lightweight BlazeFace box detector (two separate engines) |
| **Analytics** | Blink counting (Eye Aspect Ratio + hysteretic state machine), mouth open/closed (MAR), head pose yaw/pitch/roll via 6-point PnP |
| **Sources** | Webcam (auto-reconnect), video files (loop option), still images — one unified `FrameSource` API |
| **Output** | Live HUD (smoothed FPS, face count, blinks, mouth, pose), timestamped snapshots, annotated MP4 recording |
| **Engineering** | Typed frozen dataclass configs, SSRF-hardened model manager with SHA-256 pinned models, structured logging, 68 pytest tests, ruff-clean, CI on Python 3.10–3.12 |

## Install

```bash
git clone https://github.com/ALSRKAL/face_detected.git
cd face_detected
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or: pip install -e .
```

Model files are **not** stored in the repo; they download once on first use into
`~/.cache/facedetected/models` (override with `FACEDETECTED_MODELS_DIR`) and are
verified against pinned SHA-256 digests.

```bash
facedetected download-models   # optional: pre-fetch
```

## Quick start

### Live camera (default)

```bash
facedetected run                 # camera 0, interactive window
facedetected run --faces 2 --iris --record
facedetected run /path/to/video.mp4 --loop
facedetected run --headless --max-frames 300   # servers / CI: no window
facedetected run --headless --web              # watch in the browser:
                                               # http://127.0.0.1:8000
```

### Still images

```bash
facedetected image photo.jpg --json
facedetected image *.jpg --detector          # boxes only, faster
```

Annotated copies land in `outputs/`.

### Keyboard controls (live window)

| Key | Action |
|---|---|
| `q` / `ESC` | quit |
| `s` | save annotated snapshot |
| `r` | start / stop recording |
| `m` | toggle face-contour mesh |
| `b` | toggle bounding boxes |
| `i` | toggle landmark index labels (debug) |
| `h` | toggle HUD panel |

### Library API

```python
import cv2
from facedetected import DetectorConfig
from facedetected.engine import FaceLandmarkerEngine
from facedetected.analytics import FaceAnalytics
from facedetected.utils import FPSMeter

img = cv2.imread("photo.jpg")

with FaceLandmarkerEngine(DetectorConfig(num_faces=2), mode="image") as engine:
    result = engine.detect_image(img)

analytics = FaceAnalytics()
analytics.update(result)          # per-frame state
for face in result.faces:
    stats = analytics.stats_for(face.index)
    print(face.box, len(face.landmarks),
          f"blinks={stats.blink_count}", f"pose={stats.pose}")
```

More in [`examples/`](examples/).

## Architecture

```
facedetected/
├── engine/        FaceLandmarkerEngine, FaceDetectorEngine, typed results
├── analytics/     blink (EAR), mouth (MAR), head pose (solvePnP), aggregator
├── io_sources/    camera (auto-reconnect) / video (loop) / image sources
├── overlays/      mesh & box rendering, translucent HUD panel
├── models/        SSRF-hardened downloader, SHA-256 pinned MediaPipe assets
├── app.py         interactive loop, snapshots, recording, session summary
└── cli.py         facedetected run | image | download-models
```

## Development

```bash
pip install -r requirements-dev.txt
ruff check facedetected tests     # lint
pytest                            # unit tests (no network)
pytest -m ''                      # + integration tests (downloads models)
```

The historical demo screenshot [`photo.jpg`](photo.jpg) from the original
repository is kept for reference — note that it is *already mesh-overlaid*,
so the detector correctly finds no face in it.

## License

[MIT](LICENSE) © ALSRKAL

## نظرة سريعة بالعربية

مشروع **facedetected** لتحليل الوجبات البشرية في الوقت الحقيقي: شبكة 478 نقطة،
عدّ الرمشات، كشف فتح الفم، وضعية الرأس، لقطات وتسجيل فيديو — مبنٍي على
MediaPipe Tasks و OpenCV، بحزمة بايثون مهيكلة ومُختبَرة بالكامل.

```bash
facedetected run          # تشغيل مباشر على الكاميرا
facedetected image photo.jpg --json   # تحليل صورة
```

الصورة `photo.jpg` الأصلية محفوظة في المستودع كما هي لأغراض التوثيق التاريخي.
