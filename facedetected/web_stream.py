"""MJPEG web preview: watch the annotated stream and live stats in a browser.

Serves a tiny self-contained dashboard on a local HTTP port:
  ``/``           — viewer page (stream image + live stats panel)
  ``/stream.mjpg`` — multipart MJPEG of the annotated frames
  ``/stats.json``  — JSON snapshot of the per-frame analytics

Designed for the headless mode (servers, CI — or Wayland setups where
OpenCV's highgui window is unavailable).  Binds to the loopback interface
by default so the stream is never exposed to the network.
"""

from __future__ import annotations

import json
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
  body { background:#141210; color:#e8e4de; font-family: system-ui, sans-serif;
         margin:0; display:flex; flex-direction:column; align-items:center; }
  h1 { font-size:1.1rem; font-weight:600; margin:14px 0 8px; }
  img { max-width:96vw; max-height:70vh; border-radius:10px;
        border:1px solid #3a352f; background:#000; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
          gap:8px; width:min(94vw,860px); margin:12px 0 20px; }
  .card { background:#201c18; border:1px solid #3a352f; border-radius:9px;
          padding:9px 12px; }
  .k { color:#8f8578; font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; }
  .v { font-size:1.05rem; margin-top:3px; }
</style>
</head>
<body>
  <h1>facedetected — live dashboard</h1>
  <img src="/stream.mjpg" alt="annotated stream">
  <div class="grid" id="stats"></div>
<script>
async function poll() {
  try {
    const r = await fetch('/stats.json');
    const s = await r.json();
    const grid = document.getElementById('stats');
    grid.innerHTML = Object.entries(s).map(([k, v]) =>
      `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`
    ).join('');
  } catch (e) { /* server restarting; try again */ }
  setTimeout(poll, 400);
}
poll();
</script>
</body>
</html>"""


class MJPEGServer:
    """Threaded HTTP server streaming the annotated frames plus live stats.

    Usage:
        server = MJPEGServer(("127.0.0.1", 8000))
        server.start()
        ...  # per analysis frame:
        server.update_bgr(frame)        # annotated image
        server.update_stats({"FPS": "28.4", "Blinks": 3})
        ...
        server.stop()
    """

    def __init__(self, bind: tuple[str, int] = ("127.0.0.1", 8000)) -> None:
        self.bind = bind
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._stats: dict | None = None
        self._frame_seq = 0
        self._frames_sent = 0
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- frame / stats supply -------------------------------------------------

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

    def update_stats(self, stats: dict) -> None:
        """Publish the per-frame stats snapshot served at ``/stats.json``."""
        with self._condition:
            self._stats = dict(stats)

    # -- lifecycle -------------------------------------------------------------

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

    # -- request handling -------------------------------------------------------

    def _make_handler(self):
        server = self
        BOUNDARY = b"--frame\r\n"

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server API
                if self.path in ("/", "/index.html"):
                    self._serve_page()
                elif self.path == "/stream.mjpg":
                    self._serve_stream()
                elif self.path == "/stats.json":
                    self._serve_stats()
                else:
                    self.send_error(404)

            def _serve_page(self) -> None:
                body = _PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_stats(self) -> None:
                with server._condition:
                    payload = json.dumps(server._stats or {})
                body = payload.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
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
