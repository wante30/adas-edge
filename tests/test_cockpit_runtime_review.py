"""Independent runtime lifecycle checks, using no model, codec, or real window."""
from threading import Event

import numpy as np
import pytest

from tools.competition_console.runtime import BoundedVideoWriter, PrefetchedCapture
from tools import run_competition_console as runner


class Capture:
    def __init__(self, *, seek_ok=True):
        self.index = 0
        self.seek_ok = seek_ok
        self.released = False
        self.ahead = Event()
    def read(self):
        if self.index >= 3: self.ahead.set()
        value = self.index
        self.index += 1
        return True, np.full((2, 2, 3), value, np.uint8)
    def set(self, prop, value):
        if self.seek_ok: self.index = int(value)
        return self.seek_ok
    def release(self): self.released = True


def test_stop_and_seek_discard_prefetched_frames_even_with_full_queue():
    source = Capture()
    capture = PrefetchedCapture(source, capacity=1)
    assert capture.read()[1][0, 0, 0] == 0
    assert capture.read()[1][0, 0, 0] == 1
    assert source.ahead.wait(1)
    old_worker = capture.worker
    assert capture.set(0, 0)
    assert not old_worker.is_alive()
    assert [int(capture.read()[1][0, 0, 0]) for _ in range(3)] == [0, 1, 2]
    capture.release()
    assert capture.worker is None and source.released


def test_released_capture_does_not_retain_queued_images_in_exitstack_callbacks():
    source = Capture()
    capture = PrefetchedCapture(source, capacity=1)
    capture.read()
    capture.read()
    assert source.ahead.wait(1) and not capture.queue.empty()
    capture.release()
    assert capture.queue.empty()


def test_recording_applies_backpressure_instead_of_dropping_frames():
    from threading import Thread
    entered, resume, completed = Event(), Event(), Event()
    frames = []
    class SlowSink:
        def isOpened(self): return True
        def write(self, image):
            entered.set()
            assert resume.wait(2)
            frames.append(int(image[0, 0, 0]))
        def release(self): pass
    writer = BoundedVideoWriter('unused', 0, 30, (2, 2), capacity=1, writer_factory=lambda *a: SlowSink())
    writer.write(np.zeros((2, 2, 3), np.uint8))
    assert entered.wait(1)
    writer.write(np.ones((2, 2, 3), np.uint8))
    def third():
        writer.write(np.full((2, 2, 3), 2, np.uint8))
        completed.set()
    producer = Thread(target=third)
    producer.start()
    try:
        assert not completed.wait(.02)
    finally:
        resume.set()
        producer.join(2)
        writer.release()
    assert completed.is_set() and frames == [0, 1, 2]
    assert writer.summary()['peak_queued'] <= 1


@pytest.fixture
def application(monkeypatch, tmp_path):
    source = Capture()
    state = {'destroyed': False, 'writer_released': False}
    class Detector:
        def detect(self, image): return [], 0.
    class View:
        def __init__(self, *a): pass
        def render(self, *a, **kw): return np.zeros((2, 2, 3), np.uint8)
    monkeypatch.setattr(runner, 'digest', lambda path: 'test-sha')
    monkeypatch.setattr(runner, 'PcDetector', lambda *a, **kw: Detector())
    monkeypatch.setattr(runner, 'NativeView', View)
    monkeypatch.setattr(runner, 'source_capture', lambda path: (PrefetchedCapture(source), 30., 1))
    for name in ('namedWindow', 'resizeWindow', 'setMouseCallback', 'imshow'):
        monkeypatch.setattr(runner.cv2, name, lambda *a, **kw: None)
    monkeypatch.setattr(runner.cv2, 'getWindowProperty', lambda *a: 1)
    monkeypatch.setattr(runner.cv2, 'waitKey', lambda *a: -1)
    monkeypatch.setattr(runner.cv2, 'destroyAllWindows', lambda: state.update(destroyed=True))
    args = runner.parser().parse_args(['--output', str(tmp_path/'run'), '--max-frames', '1'])
    return args, source, state


def test_failed_initial_seek_is_not_labeled_as_requested_source_frame(application):
    args, source, _ = application
    source.seek_ok = False
    args.headless = True
    with pytest.raises((RuntimeError, ValueError), match='[Ss]eek'):
        runner.run(args)
    assert source.index == 0 and source.released


def test_native_window_initialization_failure_closes_capture_and_recording(application, monkeypatch):
    args, source, state = application
    class Writer:
        def __init__(self, *a, **kw): pass
        def isOpened(self): return True
        def release(self): state['writer_released'] = True
    def fail(*a, **kw): raise RuntimeError('window creation failure')
    args.record = True
    monkeypatch.setattr(runner, 'BoundedVideoWriter', Writer)
    monkeypatch.setattr(runner.cv2, 'namedWindow', fail)
    with pytest.raises(RuntimeError, match='window creation failure'):
        runner.run(args)
    assert source.released and state['writer_released']


def test_recording_flush_failure_does_not_skip_native_window_cleanup(application, monkeypatch):
    args, source, state = application
    class Writer:
        def __init__(self, *a, **kw): pass
        def isOpened(self): return True
        def write(self, image): pass
        def release(self):
            state['writer_released'] = True
            raise RuntimeError('recording flush failure')
    args.record = True
    monkeypatch.setattr(runner, 'BoundedVideoWriter', Writer)
    with pytest.raises(RuntimeError, match='recording flush failure'):
        runner.run(args)
    assert source.released and state['writer_released'] and state['destroyed']
