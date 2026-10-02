"""Frame sources: unified camera / video-file / image reading."""

from facedetected.io_sources.base import Frame, FrameSource
from facedetected.io_sources.camera import CameraSource
from facedetected.io_sources.image import ImageSource
from facedetected.io_sources.video import VideoFileSource

__all__ = ["CameraSource", "Frame", "FrameSource", "ImageSource", "VideoFileSource"]
