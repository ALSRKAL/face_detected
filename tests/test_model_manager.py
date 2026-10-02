"""Model manager: SSRF guard, caching and integrity checks (no network needed)."""

import hashlib
from unittest import mock

import pytest

from facedetected.models import manager as mgr
from facedetected.models.manager import ModelSpec, ensure_model, is_cached


@pytest.fixture
def spec(tmp_path) -> ModelSpec:
    payload = b"fake-model-bytes"
    return ModelSpec(
        key="fake",
        url="https://storage.googleapis.com/example/fake.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        filename="fake.bin",
        description="fake model for tests",
    )


class TestUrlGuard:
    def test_rejects_non_http_schemes(self):
        for url in ("ftp://example.com/x", "file:///etc/passwd", "gopher://x"):
            with pytest.raises(mgr.ModelDownloadError):
                mgr._assert_public_https_url(url)

    def test_rejects_localhost(self):
        with pytest.raises(mgr.ModelDownloadError):
            mgr._assert_public_https_url("http://localhost:8080/model.bin")
        with pytest.raises(mgr.ModelDownloadError):
            mgr._assert_public_https_url("http://sub.localhost/model.bin")

    def test_rejects_private_and_reserved_hosts(self):
        # getaddrinfo is stubbed so the test never touches the network.
        def fake_resolver(host, port):
            ip = {"10.0.0.5": "10.0.0.5", "192.168.1.1": "192.168.1.1",
                  "172.16.0.9": "172.16.0.9", "127.0.0.1": "127.0.0.1",
                  "169.254.1.1": "169.254.1.1", "240.0.0.1": "240.0.0.1",
                  "0.0.0.0": "0.0.0.0", "224.0.0.1": "224.0.0.1"}[host]
            return [(2, 1, 6, "", (ip, 0))]

        for host in ("10.0.0.5", "192.168.1.1", "172.16.0.9", "127.0.0.1",
                     "169.254.1.1", "240.0.0.1", "0.0.0.0", "224.0.0.1"):
            with mock.patch.object(mgr.socket, "getaddrinfo", side_effect=fake_resolver):
                with pytest.raises(mgr.ModelDownloadError):
                    mgr._assert_public_https_url(f"https://{host}/model.bin")

    def test_accepts_public_host(self):
        def fake_resolver(host, port):
            return [(2, 1, 6, "", ("142.250.185.208", 0))]

        with mock.patch.object(mgr.socket, "getaddrinfo", side_effect=fake_resolver):
            mgr._assert_public_https_url("https://storage.googleapis.com/model.bin")


class TestEnsureModel:
    def test_correct_file_is_cached(self, tmp_path, spec):
        target = tmp_path / spec.filename
        target.write_bytes(b"fake-model-bytes")
        path = ensure_model(spec, tmp_path)
        assert path == target
        assert is_cached(spec, tmp_path)

    def test_corrupt_cache_triggers_redownload(self, tmp_path, spec):
        target = tmp_path / spec.filename
        target.write_bytes(b"corrupted!")
        assert not is_cached(spec, tmp_path)

        def fake_download(s, destination):
            destination.write_bytes(b"fake-model-bytes")
            return destination

        with mock.patch.object(mgr, "_download", side_effect=fake_download) as dl:
            ensure_model(spec, tmp_path)
            dl.assert_called_once()
        assert target.read_bytes() == b"fake-model-bytes"

    def test_bad_download_rejected_and_removed(self, tmp_path, spec):
        def bad_download(s, destination):
            destination.write_bytes(b"tampered-payload")
            return destination

        with mock.patch.object(mgr, "_download", side_effect=bad_download):
            with pytest.raises(mgr.ModelIntegrityError):
                ensure_model(spec, tmp_path, force=True)
        assert not (tmp_path / spec.filename).exists()

    def test_force_redownloads_valid_cache(self, tmp_path, spec):
        target = tmp_path / spec.filename
        target.write_bytes(b"fake-model-bytes")
        with mock.patch.object(mgr, "_download", return_value=target) as dl:
            ensure_model(spec, tmp_path, force=True)
            dl.assert_called_once()

    def test_default_cache_dir_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FACEDETECTED_MODELS_DIR", str(tmp_path / "models"))
        assert mgr.default_cache_dir() == tmp_path / "models"
