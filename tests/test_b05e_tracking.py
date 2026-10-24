import cv2
import numpy as np
from pathlib import Path

from competition_ai.traffic_light_tracking import (
    TrafficLightTracker,
    track_bbox_lk,
)


def _textured_frame(shift_x=0):
    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    # Realistic enough texture for LK: a lamp plus surrounding edges.
    x1, y1, x2, y2 = 80 + shift_x, 60, 120 + shift_x, 115
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 12, (0, 220, 0), -1)
    for x in range(0, 240, 8):
        cv2.line(frame, (x, 0), (x, 179), (70, 70, 70), 1)
    return frame


def _det(bbox):
    return {"class_name": "traffic light", "bbox": bbox, "confidence": 0.9}


def test_lk_tracks_real_texture_shift():
    a = _textured_frame(0)
    b = _textured_frame(12)
    box = track_bbox_lk(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(b, cv2.COLOR_BGR2GRAY),
                        (80, 60, 120, 115))
    assert box is not None
    assert box[0] > 90


def test_update_frame_uses_new_frame_and_duplicate_is_ignored():
    tracker = TrafficLightTracker(detection_ttl_ms=1200, color_ttl_ms=500)
    a = _textured_frame(0)
    b = _textured_frame(10)
    tracker.update_detections(a, [_det((80, 60, 120, 115))], source_frame_id=10,
                              timestamp_ns=1_000_000_000)
    assert tracker.update_frame(a, source_frame_id=10, timestamp_ns=1_000_000_000) == []
    rows = tracker.update_frame(b, source_frame_id=11, timestamp_ns=1_100_000_000)
    assert len(rows) == 1
    assert rows[0]["source_frame_id"] == 11
    assert rows[0]["track_valid"] is True
    # The duplicate call must not increase colour confirmation.
    assert tracker.update_frame(b, source_frame_id=11, timestamp_ns=1_100_000_000) == []


def test_generation_change_resets_tracks():
    tracker = TrafficLightTracker()
    a = _textured_frame(0)
    tracker.update_detections(a, [_det((80, 60, 120, 115))], source_frame_id=1,
                              timestamp_ns=1_000_000_000, generation=7)
    assert len(tracker.update_frame(_textured_frame(3), source_frame_id=2,
                                    timestamp_ns=1_100_000_000, generation=7)) == 1
    assert tracker.update_frame(_textured_frame(3), source_frame_id=3,
                                timestamp_ns=1_200_000_000, generation=8) == []


def test_multiple_tracks_and_lost_tracking():
    tracker = TrafficLightTracker()
    a = _textured_frame(0)
    detections = [_det((30, 30, 70, 80)), _det((150, 40, 190, 95))]
    rows = tracker.update_detections(a, detections, source_frame_id=1,
                                     timestamp_ns=1_000_000_000)
    assert len(rows) == 2
    # A featureless frame cannot track either ROI.
    blank = np.zeros_like(a)
    rows = tracker.update_frame(blank, source_frame_id=2, timestamp_ns=1_100_000_000)
    assert len(rows) == 2
    assert all(not row["track_valid"] for row in rows)


def test_500ms_colour_and_1200ms_detection_expiry():
    tracker = TrafficLightTracker(detection_ttl_ms=1200, color_ttl_ms=500)
    a = _textured_frame(0)
    tracker.update_detections(a, [_det((80, 60, 120, 115))], source_frame_id=1,
                              timestamp_ns=1_000_000_000)
    rows = tracker.update_frame(_textured_frame(2), source_frame_id=2,
                                timestamp_ns=1_100_000_000)
    assert rows[0]["detection_age_ms"] == 100.0
    rows = tracker.current(1_700_000_000)
    assert rows[0]["color_age_ms"] >= 500.0
    assert rows[0]["color_valid"] is False
    rows = tracker.current(2_300_000_000)
    assert rows == []


def test_real_video_light_roi_tracks_and_stays_green():
    root = Path(__file__).parent / "data"
    a = cv2.imread(str(root / "b05e_real_light_0.png"))
    b = cv2.imread(str(root / "b05e_real_light_1.png"))
    assert a is not None and b is not None
    box = track_bbox_lk(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(b, cv2.COLOR_BGR2GRAY),
                        (40, 35, 140, 92))
    assert box is not None
    assert box[0] > 40
    from competition_ai.traffic_light_tracking import classify_lamp_color
    assert classify_lamp_color(a, (40, 35, 140, 92)).color.value == "GREEN"
    assert classify_lamp_color(b, box).color.value == "GREEN"
