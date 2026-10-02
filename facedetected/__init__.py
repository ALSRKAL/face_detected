"""facedetected — professional real-time face analysis toolkit.

Built on MediaPipe Tasks (FaceLandmarker / FaceDetector) and OpenCV.
"""

from facedetected.config import DetectorConfig, HudConfig, OverlayConfig, SourceConfig

__version__ = "2.0.0"

__all__ = [
    "DetectorConfig",
    "HudConfig",
    "OverlayConfig",
    "SourceConfig",
    "__version__",
]
