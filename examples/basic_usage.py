"""Minimal library example: detect faces in a still image.

Run:  python examples/basic_usage.py [path/to/image.jpg]
"""

import sys

import cv2

from facedetected import DetectorConfig, OverlayConfig
from facedetected.engine import FaceLandmarkerEngine
from facedetected.overlays.draw import draw_result

path = sys.argv[1] if len(sys.argv) > 1 else "photo.jpg"
img = cv2.imread(path)
if img is None:
    sys.exit(f"cannot read image: {path}")

with FaceLandmarkerEngine(DetectorConfig(num_faces=5), mode="image") as engine:
    result = engine.detect_image(img)

draw_result(img, result, OverlayConfig())
print(f"faces: {result.face_count}")
for face in result.faces:
    print(f"  face {face.index}: box={face.box}, landmarks={len(face.landmarks)}")

out = "outputs/annotated.jpg"
cv2.imwrite(out, img)
print(f"annotated image written to {out}")
