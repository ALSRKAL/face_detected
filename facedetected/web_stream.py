"""MJPEG web preview: watch the annotated stream from any browser.

Serves a tiny self-contained page plus a multipart MJPEG stream on a local
HTTP port.  Designed for the headless mode (servers, CI — or Wayland setups
where OpenCV's highgui window is unavailable).  Binds to the loopback
interface by default so the stream is never exposed to the network.
"""

from __future__ import annotations

import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>facedetected live</title>
<style>
  body {{ background:#141210; color:#e8e4de; font-family: system-ui, sans-serif;
         margin:0; display:flex; flex-direction:column; align-items:center; }}
  h1 {{ font-size:1.1rem; font-weight:600; margin:14px 0 6px; }}
  img {{ max-width:96vw; max-height:82vh; border-radius:10px;
        border:1px solid #3a352f; background:#000; }}
  .stats {{ font-size:.9rem; color:#b5aa9c; margin:8px 0 14px; }}
</style>
</head>
<body>
  <h1>facedetected — live stream</h1>
  <img src="/stream.mjpg" alt="annotated stream">
  <div class="stats">annotated output of the analysis loop</div>
</body>
</html>"""


class MJPEGServer:
    """Threaded HTTP server streaming the latest annotated frame as MJPEG.

    Usage:
        server = MJPEGServer(("127.0.0.1", 8000))
        server.start()
        ...  # call server.update_bgr(frame) once per analysis frame
        server.stop()
    """

    def __init__(self, bind: tuple[str, int] = ("127.0.0.1", 8000)) -> None:
        self.bind = bind
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._frame_seq = 0
        self._frames_sent = 0
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- frame supply --------------------------------------------------------

    def update_bgr(self, frame_bgr: np.ndarray) -> None:
        """Encode one BGR frame and publish it to connected viewers."""
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            self.update(buf.tobytes())

    def update(self, jpeg_bytes: bytes) -> None:
        with self._condition:
            self._jpeg = jpeg_bytes
            self._frame_seq += 1
            self._condition.notify_all()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> tuple[str, int]:
        """Bind and serve in a daemon thread; returns the actual (host, port)."""
        if self._httpd is not None:
            return self.bind
        self._httpd = ThreadingHTTPServer(self.bind, self._make_handler())
        host, port = self._httpd.server_address[:2]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="facedetected-mjpeg", daemon=True
        )
        self._thread.start()
        logger.info("web preview ready: http://%s:%s", host, port)
        return str(host), int(port)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    @property
    def url(self) -> str:
        host, port = self.bind
        return f"http://{host}:{port}"

    @property
    def frames_sent(self) -> int:
        return self._frames_sent

    # -- request handling ----------------------------------------------------

    def _make_handler(self):
        server = self
        BOUNDARY = b"--frame\r\n"

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server API
                if self.path in ("/", "/index.html"):
                    self._serve_page()
                elif self.path == "/stream.mjpg":
                    self._serve_stream()
                else:
                    self.send_error(404)

            def _serve_page(self) -> None:
                body = _PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_stream(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last_sent = 0
                try:
                    while True:
                        with server._condition:
                            if server._frame_seq == last_sent:
                                server._condition.wait(timeout=1.0)
                            if server._frame_seq == last_sent:
                                continue  # no new frame; loop back and wait
                            last_sent = server._frame_seq
                            jpeg = server._jpeg
                        server._frames_sent += 1
                        self.wfile.write(
                            BOUNDARY
                            + b"Content-Type: image/jpeg\r\n"
                            + b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                            + jpeg + b"\r\n"
                        )
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return  # viewer disconnected or server shutting down

            def log_message(self, *args) -> None:  # silence per-request logs
                logger.debug("http: %s", args[0] if args else "")

        return Handler
