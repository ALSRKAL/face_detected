"""MJPEG web preview server (local loopback only, ephemeral port)."""

from __future__ import annotations

from http.client import HTTPConnection

import numpy as np
import pytest

from facedetected.web_stream import MJPEGServer


@pytest.fixture
def server():
    server = MJPEGServer(("127.0.0.1", 0))
    host, port = server.start()
    yield server, host, port
    server.stop()


def _frame() -> np.ndarray:
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    frame[:, :, 0] = 200  # visible blue-ish block
    return frame


def _get(host: str, port: int, path: str):
    conn = HTTPConnection(host, port, timeout=5)
    conn.request("GET", path)
    response = conn.getresponse()
    return conn, response


class TestMJPEGServer:
    def test_index_page(self, server):
        server, host, port = server
        conn, response = _get(host, port, "/")
        assert response.status == 200
        assert response.headers["Content-Type"].startswith("text/html")
        body = response.read()
        conn.close()
        assert b"stream.mjpg" in body

    def test_mjpeg_stream_delivers_frames(self, server):
        server, host, port = server
        server.update_bgr(_frame())
        server.update_bgr(_frame())
        conn, response = _get(host, port, "/stream.mjpg")
        assert response.status == 200
        assert "multipart/x-mixed-replace" in response.headers["Content-Type"]
        chunk = response.read1(65536)
        conn.close()
        assert chunk.startswith(b"--frame\r\n")
        assert b"Content-Type: image/jpeg\r\n" in chunk
        assert server.frames_sent >= 1

    def test_unknown_path_404(self, server):
        server, host, port = server
        conn, response = _get(host, port, "/nope")
        assert response.status == 404
        conn.close()

    def test_update_before_start_is_harmless(self):
        server = MJPEGServer(("127.0.0.1", 0))
        server.update_bgr(_frame())  # no server thread yet: just buffers
        assert server._frame_seq == 1
