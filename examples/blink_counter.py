"""Blink counting over a video file using the library API.

Run:  python examples/blink_counter.py path/to/video.mp4
"""

import sys

from facedetected import DetectorConfig
from facedetected.analytics import FaceAnalytics
from facedetected.engine import FaceLandmarkerEngine
from facedetected.io_sources import VideoFileSource

if len(sys.argv) < 2:
    sys.exit("usage: python examples/blink_counter.py <video.mp4>")

source = VideoFileSource(sys.argv[1])
engine = FaceLandmarkerEngine(DetectorConfig(num_faces=1), mode="video")
analytics = FaceAnalytics()

frames = 0
with source, engine:
    while True:
        frame = source.read()
        if frame is None:
            break
        result = engine.detect_video(frame.image, frame.timestamp_ms)
        analytics.update(result)
        frames += 1

print(f"processed {frames} frames")
print(f"total blinks: {analytics.total_blinks()}")
for index in sorted({f.index for f in analytics._trackers}):
    print(f"  face {index}: {analytics.total_blinks(index)} blinks")
