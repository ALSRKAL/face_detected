"""FPSMeter behaviour."""

import pytest

from facedetected.utils.fps import FPSMeter


class TestFPSMeter:
    def test_first_tick_returns_zero(self):
        meter = FPSMeter()
        assert meter.tick(now=100.0) == 0.0

    def test_constant_interval_converges(self):
        meter = FPSMeter(smoothing=0.0)  # no smoothing: instant convergence
        meter.tick(now=0.0)
        fps = meter.tick(now=0.05)  # 50 ms -> 20 fps
        assert fps == pytest.approx(20.0, rel=1e-3)

    def test_smoothing_blends(self):
        meter = FPSMeter(smoothing=0.5)
        meter.tick(now=0.0)
        meter.tick(now=0.1)   # delta 0.1 -> avg 0.1 -> 10 fps
        meter.tick(now=0.2)   # delta 0.1 -> avg 0.1 -> 10 fps
        assert meter.fps == pytest.approx(10.0, rel=1e-3)

    def test_frame_count_and_reset(self):
        meter = FPSMeter()
        for i in range(5):
            meter.tick(now=float(i))
        assert meter.frame_count == 5
        meter.reset()
        assert meter.frame_count == 0
        assert meter.fps == 0.0

    def test_rejects_bad_parameters(self):
        with pytest.raises(ValueError):
            FPSMeter(history=0)
        with pytest.raises(ValueError):
            FPSMeter(smoothing=1.0)
