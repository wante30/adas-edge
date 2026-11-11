import numpy as np
import cv2

from competition_ai.multi_signal import (
    Attributes, Candidate, HSVBackend, MultiSignalTracker, q4_box,
)


def scene():
    image = np.zeros((100, 160, 3), np.uint8)
    cv2.circle(image, (30, 30), 10, (0, 255, 0), -1)
    cv2.circle(image, (90, 30), 10, (0, 0, 255), -1)
    return image


def candidates(fid, ns):
    return [Candidate((15, 15, 45, 45), .9, fid, ns),
            Candidate((75, 15, 105, 45), .8, fid, ns)]


def test_independent_colors_and_no_direction_from_position():
    tracker = MultiSignalTracker()
    for fid in range(2):
        ns = 1_000_000_000+fid*500_000_000
        tracker.update(scene(), candidates(fid, ns), frame_id=fid, timestamp_ns=ns, context=("FAST-Q4", 1))
    result = tracker.snapshot(ns)
    assert [s['color'] for s in result['signals']] == ['GREEN', 'RED']
    assert all(s['ego_relevant'] == 'UNKNOWN' for s in result['signals'])
    assert all(d['color'] == 'UNKNOWN' for d in result['directions'].values())


def test_duplicate_models_do_not_confirm_twice_or_change_id():
    tracker = MultiSignalTracker()
    for fid in range(2):
        ns = 1_000_000_000+fid*500_000_000
        row = candidates(fid, ns)[0]
        dup = Candidate(row.bbox, .7, fid, ns, 'specialist')
        tracker.update(scene(), [row, dup], frame_id=fid, timestamp_ns=ns, context=(1,))
        rows = tracker.snapshot(ns)['signals']
        assert len(rows) == 1 and rows[0]['track_id'] == 1
        assert rows[0]['detectors'] == ['main', 'specialist']
        assert rows[0]['color'] == ('UNKNOWN' if fid == 0 else 'GREEN')


def test_no_refresh_from_historical_or_repeated_detection_and_ttl():
    tracker = MultiSignalTracker(confirm_updates=1)
    tracker.update(scene(), candidates(1, 1_000_000_000), frame_id=1, timestamp_ns=1_000_000_000, context=(1,))
    assert not tracker.update(scene(), candidates(1, 1_000_000_000), frame_id=1, timestamp_ns=1_000_000_000, context=(1,))
    tracker.update(scene(), candidates(1, 1_000_000_000), frame_id=2, timestamp_ns=1_500_000_000, context=(1,))
    out = tracker.snapshot(1_500_000_000)
    assert out['rejections'] == ['DETECTION_FRAME_MISMATCH']*2
    assert all(s['color'] == 'UNKNOWN' and not s['valid'] for s in out['signals'])
    assert all(s['reason'] == 'EXPIRED' for s in tracker.snapshot(2_200_000_000)['signals'])


def test_model_epoch_and_disconnect_invalidate():
    tracker = MultiSignalTracker(confirm_updates=1)
    tracker.update(scene(), candidates(2, 1_000_000_000), frame_id=2, timestamp_ns=1_000_000_000, context=(1,))
    assert tracker.snapshot(1_100_000_000, context=(2,))['signals'] == []
    assert all(not s['valid'] and s['color'] == 'UNKNOWN'
               for s in tracker.snapshot(1_100_000_000, available=False)['signals'])
    tracker.update(scene(), candidates(1, 1_200_000_000), frame_id=1, timestamp_ns=1_200_000_000, context=(2,))
    assert all(s['track_id'] > 2 for s in tracker.snapshot(1_200_000_000)['signals'])


def test_coordinate_spaces_and_mosaic_seam():
    assert q4_box((650, 370, 660, 390), space='mosaic', width=640, height=360) == (10, 10, 20, 30)
    assert q4_box((20, 20, 40, 60), space='q4_display', width=640, height=360) == (10, 10, 20, 30)
    for box in [(630, 370, 660, 390), (10, 10, 20, 30), (650, 350, 660, 390), (650, 370, float('nan'), 390)]:
        assert q4_box(box, space='mosaic', width=640, height=360) is None


def test_mixed_glow_small_and_dim_are_unknown():
    backend = HSVBackend()
    assert backend.classify(np.zeros((3, 3, 3), np.uint8)).color == 'UNKNOWN'
    assert backend.classify(np.full((24, 24, 3), 50, np.uint8)).color == 'UNKNOWN'
    assert backend.classify(scene()[10:50, 10:110]).color == 'UNKNOWN'


def test_changed_color_immediately_removes_old_green():
    tracker = MultiSignalTracker()
    for fid in range(3):
        ns = 1_000_000_000+fid*500_000_000
        image = scene()
        if fid == 2:
            image[:, :, [1, 2]] = image[:, :, [2, 1]]
        tracker.update(image, candidates(fid, ns), frame_id=fid, timestamp_ns=ns, context=(1,))
    assert all(s['color'] == 'UNKNOWN' for s in tracker.snapshot(ns)['signals'])
