import time
import threading

from competition_ai.multi_signal import Attributes, MultiSignalTracker, Candidate
from competition_ai.multi_signal_worker import LatestSignalWorker, expire_result
from competition_ai.signal_fpga_backend import FPGAAttributeBackend
from test_multi_signal import scene, candidates


def test_fpga_attributes_require_exact_frame_bbox_epoch_and_finite_scores():
    request = dict(frame_id=3, timestamp_ns=10, bbox=[1, 2, 8, 9], context=['epoch', 1], source_id='Q4')
    response = dict(request, color='RED', shape='LEFT_ARROW', color_confidence=.99, shape_confidence=.92)
    backend = FPGAAttributeBackend(lambda identity: response)
    assert backend.classify(scene(), request=request).shape == 'LEFT_ARROW'
    response['frame_id'] = 2
    assert backend.classify(scene(), request=request).color == 'UNKNOWN'
    response['frame_id'] = 3
    response['shape_confidence'] = float('nan')
    assert backend.classify(scene(), request=request).shape == 'UNKNOWN'


def test_color_stability_is_independent_of_shape_changes():
    class Backend:
        index = 0
        def classify(self, roi, *, request=None):
            self.index += 1
            return Attributes('GREEN', 'LEFT_ARROW' if self.index % 2 else 'UNKNOWN', .9, .9, 'OK')
    tracker = MultiSignalTracker(backend=Backend())
    for fid in range(3):
        ns = 1_000_000_000+fid*500_000_000
        tracker.update(scene(), candidates(fid, ns)[:1], frame_id=fid, timestamp_ns=ns, context=(1,))
    signal = tracker.snapshot(ns)['signals'][0]
    assert signal['color'] == 'GREEN' and signal['shape'] == 'UNKNOWN'


def test_latest_task_overwrites_and_old_context_completion_is_rejected():
    entered, release = threading.Event(), threading.Event()
    class Backend:
        def classify(self, roi, *, request=None):
            entered.set()
            assert release.wait(2)
            return Attributes('GREEN', 'LEFT_ARROW', .9, .9, 'OK')
    worker = LatestSignalWorker(tracker=MultiSignalTracker(backend=Backend(), confirm_updates=1))
    ns = time.monotonic_ns()
    worker.submit(scene(), candidates(1, ns)[:1], frame_id=1, timestamp_ns=ns, context=(1,))
    assert entered.wait(1)
    ns = time.monotonic_ns()
    worker.submit(scene(), candidates(2, ns), frame_id=2, timestamp_ns=ns, context=(2,))
    worker.submit(scene(), candidates(3, ns), frame_id=3, timestamp_ns=ns, context=(2,))
    assert worker.replaced == 1
    assert worker.current(ns, context=(2,))['signals'] == []
    release.set()
    worker.close()
    assert not worker.thread.is_alive()
    assert worker.current(time.monotonic_ns(), context=(2,))['signals'] == []


def test_reader_expiry_does_not_extend_capture_time_or_keep_green():
    tracker = MultiSignalTracker(confirm_updates=1)
    tracker.update(scene(), candidates(1, 1_000_000_000), frame_id=1, timestamp_ns=1_000_000_000, context=(1,))
    snapshot = tracker.snapshot(1_000_000_000)
    expired = expire_result(snapshot, 2_200_000_000)
    assert expired['signals'][0]['timestamp_ns'] == 1_000_000_000
    assert expired['signals'][0]['color'] == 'UNKNOWN'
