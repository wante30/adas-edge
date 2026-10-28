import time

import numpy as np
import pytest

from tools.competition_console.pgv_capture import PgvLatestCapture, PgvLiveCapture


def test_pgv_live_capture_uses_source_identity():
    capture = PgvLiveCapture("unused", timeout_s=0.01)
    capture.source.read_latest = lambda: object.__new__(type("Frame", (), {
        "frame_id": 42, "timestamp_ns": 123, "age_ms": 7.5, "image": "image"}))

    ok, image = capture.read()

    assert ok and image == "image"
    assert (capture.frame_id, capture.timestamp_ns, capture.age_ms) == (42, 123, 7.5)
    capture.release()


def test_pgv_live_capture_times_out_without_a_new_frame():
    capture = PgvLiveCapture("unused", timeout_s=0.01)
    capture.source.read_latest = lambda: None

    start = time.monotonic()
    ok, image = capture.read()

    assert ok is False and image is None
    assert time.monotonic() - start < 1
    assert capture.error is None
    capture.release()


def test_pgv_parser_rejects_invalid_timeout():
    with pytest.raises(ValueError):
        PgvLiveCapture("unused", timeout_s=0)


def test_runner_accepts_live_pgv_source(monkeypatch, tmp_path):
    from tools import run_competition_console as runner

    class FakeLive:
        def __init__(self, path, timeout_s=5.):
            self.path, self.timeout_s = path, timeout_s
            self.frame_id, self.timestamp_ns = 9, 900
            self.error = None
            self.dropped_frames = 0

        def read(self):
            return True, np.zeros((2, 2, 3), np.uint8)

        def set(self, prop, value): return True

        def release(self): pass

    args = runner.parser().parse_args(
        ["--pgv", "latest.pgv", "--pgv-timeout", "2", "--headless", "--max-frames", "1",
         "--output", str(tmp_path / "run")])
    monkeypatch.setattr(
        runner,
        "pgv_capture",
        lambda path, timeout, *_args: (FakeLive(path, timeout), 30., None),
    )
    monkeypatch.setattr(runner.PcDetector, "detect", lambda self, image: ([], 0.))
    report = runner.run(args)

    assert str(report["video"]).endswith("latest.pgv")
