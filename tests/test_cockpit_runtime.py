from threading import Event

import numpy as np
import pytest

from tools.competition_console.runtime import BoundedVideoWriter, LatestFrameWorker, PlaybackPacer, PrefetchedCapture


def test_slow_frame_is_repaid_by_later_fast_frames_without_skipping():
    now = [0.]
    pace = PlaybackPacer(30, clock=lambda: now[0])
    assert pace.delay_ms(0) == 33
    now[0] = .080
    assert pace.delay_ms(1) == 1
    now[0] = .085
    assert 14 <= pace.delay_ms(2) <= 15
    now[0] = .105
    assert 27 <= pace.delay_ms(3) <= 28
    assert pace.summary()['last_ms'] < 6


def test_pause_and_replay_do_not_carry_old_wall_clock_debt():
    now = [0.]
    pace = PlaybackPacer(30, clock=lambda: now[0])
    pace.delay_ms(100)
    now[0] = 100.
    assert pace.delay_ms(100, paused=True) == 30
    assert 32 <= pace.delay_ms(101) <= 33
    now[0] = 100.1
    assert 32 <= pace.delay_ms(0) <= 33
    assert pace.summary()['last_ms'] == 0


class MemoryWriter:
    def __init__(self, *args):
        self.frames = []
        self.released = False
    def isOpened(self): return True
    def write(self, image): self.frames.append(image.copy())
    def release(self): self.released = True


def test_recording_keeps_all_frames_in_order_and_owns_a_copy():
    sink = MemoryWriter()
    writer = BoundedVideoWriter('unused', 0, 30, (8, 8), capacity=2, writer_factory=lambda *a: sink)
    image = np.zeros((8, 8, 3), np.uint8)
    for i in range(20):
        image.fill(i)
        writer.write(image)
        image.fill(255)
    writer.release()
    writer.release()
    assert [int(frame[0, 0, 0]) for frame in sink.frames] == list(range(20))
    assert writer.summary()['submitted_frames_encoded'] == 20
    assert writer.summary()['peak_queued'] <= 2
    assert sink.released
    with pytest.raises(RuntimeError, match='closed'): writer.write(image)


def test_encoder_failure_is_reported_even_on_final_frame():
    class Broken(MemoryWriter):
        def write(self, image): raise OSError('disk failure')
    writer = BoundedVideoWriter('unused', 0, 30, (8, 8), writer_factory=Broken)
    writer.write(np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(RuntimeError, match='encoder failed'): writer.release()


def test_encoder_open_failure_is_not_silently_ignored():
    class Broken(MemoryWriter):
        def isOpened(self): return False
    with pytest.raises(RuntimeError, match='encoder failed'):
        BoundedVideoWriter('unused', 0, 30, (8, 8), writer_factory=Broken)


def test_encoder_release_failure_is_reported():
    class Broken(MemoryWriter):
        def release(self): raise OSError('disk flush failure')
    writer = BoundedVideoWriter('unused', 0, 30, (8, 8), writer_factory=Broken)
    with pytest.raises(RuntimeError, match='encoder failed'): writer.release()


class NumberedCapture:
    def __init__(self): self.index, self.released = 0, False
    def isOpened(self): return not self.released
    def get(self, prop): return 30
    def read(self):
        if self.index >= 20: return False, None
        index = self.index
        self.index += 1
        return True, np.full((2, 2, 3), index, np.uint8)
    def set(self, prop, value): self.index = int(value); return True
    def release(self): self.released = True


def test_prefetch_preserves_frames_and_discards_old_queue_on_seek():
    source = NumberedCapture()
    capture = PrefetchedCapture(source, capacity=2)
    assert capture.get(0) == 30
    assert int(capture.read()[1][0, 0, 0]) == 0
    assert int(capture.read()[1][0, 0, 0]) == 1
    assert capture.set(0, 10)
    assert [int(capture.read()[1][0, 0, 0]) for _ in range(10)] == list(range(10, 20))
    assert capture.read() == (False, None)
    assert capture.set(0, 0)
    assert int(capture.read()[1][0, 0, 0]) == 0
    capture.release()
    capture.release()
    assert source.released


def test_prefetch_propagates_decode_errors():
    class Broken(NumberedCapture):
        def read(self): raise OSError('decode failure')
    capture = PrefetchedCapture(Broken())
    with pytest.raises(RuntimeError, match='decoding failed'): capture.read()
    capture.release()


def test_latest_frame_worker_does_not_block_or_queue_stale_work():
    started, release, finished = Event(), Event(), Event()
    calls = []

    def slow(value):
        started.set()
        assert release.wait(2)
        calls.append(value)
        finished.set()
        return value * 10

    worker = LatestFrameWorker(slow)
    try:
        assert worker.submit(1)
        assert started.wait(1)
        assert not worker.submit(2)
        assert worker.poll() is None
        release.set()
        assert finished.wait(1)
        assert worker.poll() == 10
        assert worker.submit(3)
        assert worker.wait() == 30
        assert calls == [1, 3]
    finally:
        release.set()
        worker.close()
