import cv2
import numpy as np

from competition_ai.traffic_light_async import AsyncTrafficLightTracker


def _frame(shift=0):
    f = np.zeros((180, 240, 3), np.uint8)
    x1, y1, x2, y2 = 80 + shift, 60, 120 + shift, 115
    cv2.rectangle(f, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.circle(f, ((x1 + x2) // 2, (y1 + y2) // 2), 12, (0, 220, 0), -1)
    for x in range(0, 240, 8):
        cv2.line(f, (x, 0), (x, 179), (70, 70, 70), 1)
    return f


def _det(bbox, conf=0.9):
    return {"class_name": "traffic light", "bbox": bbox, "confidence": conf}


def test_late_detection_is_propagated_to_current_frame():
    tr = AsyncTrafficLightTracker()
    for i in range(6):
        tr.push_frame(_frame(i), source_frame_id=100 + i, timestamp_ns=1_000_000_000 + i * 33_000_000,
                      generation=1)
    out = tr.update_detections([_det((80, 60, 120, 115))], detection_frame_id=100,
                               detection_timestamp_ns=1_000_000_000,
                               current_frame_id=105, current_timestamp_ns=1_165_000_000,
                               generation=1)
    assert not out["rejected"]
    assert len(out["records"]) == 1
    assert out["records"][0]["detection_frame_id"] == 100
    assert out["records"][0]["source_frame_id"] == 105
    assert out["records"][0]["bbox"][0] >= 80


def test_two_detections_do_not_match_same_track():
    tr = AsyncTrafficLightTracker()
    tr.push_frame(_frame(0), source_frame_id=1, timestamp_ns=1_000_000_000, generation=1)
    out = tr.update_detections(
        [_det((80, 60, 120, 115)), _det((80, 60, 120, 115))],
        detection_frame_id=1, detection_timestamp_ns=1_000_000_000,
        current_frame_id=1, current_timestamp_ns=1_000_000_000, generation=1,
    )
    assert len({r["track_id"] for r in out["records"]}) == 2


def test_late_old_detection_cannot_override_newer_track():
    tr = AsyncTrafficLightTracker()
    for i in range(8):
        tr.push_frame(_frame(i), source_frame_id=i, timestamp_ns=1_000_000_000 + i * 33_000_000,
                      generation=1)
    tr.update_detections([_det((80, 60, 120, 115))], detection_frame_id=5,
                         detection_timestamp_ns=1_000_000_000 + 5 * 33_000_000,
                         current_frame_id=7, current_timestamp_ns=1_000_000_000 + 7 * 33_000_000,
                         generation=1)
    out = tr.update_detections([_det((80, 60, 120, 115))], detection_frame_id=4,
                               detection_timestamp_ns=1_000_000_000 + 4 * 33_000_000,
                               current_frame_id=7, current_timestamp_ns=1_000_000_000 + 7 * 33_000_000,
                               generation=1)
    assert out["rejected"]
    assert out["records"][0]["detection_frame_id"] == 5


def test_source_generation_reset_and_duplicate_frame():
    tr = AsyncTrafficLightTracker()
    assert tr.push_frame(_frame(), source_frame_id=1, timestamp_ns=1_000_000_000,
                         generation=1, source_id="a")
    assert not tr.push_frame(_frame(), source_frame_id=1, timestamp_ns=1_000_000_001,
                             generation=1, source_id="a")
    assert tr.push_frame(_frame(), source_frame_id=2, timestamp_ns=1_000_000_002,
                         generation=2, source_id="a")
    assert tr.push_frame(_frame(), source_frame_id=3, timestamp_ns=1_000_000_003,
                         generation=2, source_id="b")


def test_small_roi_and_effective_color_fallback():
    tr = AsyncTrafficLightTracker(color_ttl_ms=500)
    tr.push_frame(_frame(), source_frame_id=1, timestamp_ns=1_000_000_000, generation=1)
    out = tr.update_detections([_det((80, 60, 120, 115))], detection_frame_id=1,
                               detection_timestamp_ns=1_000_000_000,
                               current_frame_id=1, current_timestamp_ns=1_000_000_000,
                               generation=1)
    assert out["records"][0]["effective_color"] == "UNKNOWN"
    # A blank frame cannot track the ROI; effective colour must not inherit.
    blank = np.zeros_like(_frame())
    out = tr.update_frame(blank, source_frame_id=2, timestamp_ns=1_100_000_000,
                          generation=1)
    assert out[0]["track_valid"] is False
    assert out[0]["effective_color"] == "UNKNOWN"
