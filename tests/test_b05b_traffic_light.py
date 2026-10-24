import cv2
import numpy as np

from competition_ai.traffic_light import TrafficLightState
from competition_ai.traffic_light_tracking import (
    TrafficLightTracker,
    classify_lamp_color,
)


def _lamp_frame(hue: int) -> np.ndarray:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    hsv = np.zeros((30, 30, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = 255
    frame[35:65, 45:75] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame


def test_red_yellow_green_are_distinguished():
    assert classify_lamp_color(_lamp_frame(0), (45, 35, 75, 65)).color == TrafficLightState.RED
    assert classify_lamp_color(_lamp_frame(28), (45, 35, 75, 65)).color == TrafficLightState.YELLOW
    assert classify_lamp_color(_lamp_frame(60), (45, 35, 75, 65)).color == TrafficLightState.GREEN


def test_black_or_blue_sign_is_unknown():
    black = np.zeros((100, 120, 3), dtype=np.uint8)
    assert classify_lamp_color(black, (45, 35, 75, 65)).color == TrafficLightState.UNKNOWN
    blue = np.zeros((100, 120, 3), dtype=np.uint8)
    blue[:, :] = (255, 0, 0)  # BGR blue; not an R/Y/G lamp hue
    assert classify_lamp_color(blue, (45, 35, 75, 65)).color == TrafficLightState.UNKNOWN


def _detection(x1=45, y1=35, x2=75, y2=65, confidence=0.9):
    return {"class_name": "traffic light", "bbox": [x1, y1, x2, y2], "confidence": confidence}


def test_two_lights_are_kept_separately_and_color_needs_two_updates():
    tracker = TrafficLightTracker(ttl_ms=1200)
    frame = _lamp_frame(0)
    first = tracker.update(frame, [_detection()], source_id="pgv", source_frame_id=10, timestamp_ns=1_000_000_000)
    assert len(first) == 1
    assert first[0]["color"] == "UNKNOWN"
    second = tracker.update(frame, [_detection()], source_id="pgv", source_frame_id=11, timestamp_ns=1_100_000_000)
    assert second[0]["color"] == "RED"
    assert second[0]["valid"] is True
    assert second[0]["ttl_ms"] == 1200


def test_two_lights_keep_two_records():
    tracker = TrafficLightTracker()
    frame = _lamp_frame(60)
    records = tracker.update(
        frame,
        [_detection(5, 35, 35, 65), _detection(85, 35, 115, 65)],
        source_id="pgv",
        source_frame_id=20,
        timestamp_ns=2_000_000_000,
    )
    assert len(records) == 2
    assert {record["track_id"] for record in records} == {1, 2}


def test_expired_track_is_dropped():
    tracker = TrafficLightTracker(ttl_ms=1000)
    frame = _lamp_frame(60)
    tracker.update(frame, [_detection()], source_id="pgv", source_frame_id=1, timestamp_ns=1_000_000_000)
    tracker.update(frame, [_detection()], source_id="pgv", source_frame_id=2, timestamp_ns=1_100_000_000)
    records = tracker.update(frame, [], source_id="pgv", source_frame_id=3, timestamp_ns=2_500_000_000)
    assert records == []


def test_new_red_does_not_keep_old_green_valid():
    tracker = TrafficLightTracker(ttl_ms=1200, color_ttl_ms=500)
    green = _lamp_frame(60)
    red = _lamp_frame(0)
    tracker.update(green, [_detection()], source_id="pgv", source_frame_id=1, timestamp_ns=1_000_000_000)
    rows = tracker.update(green, [_detection()], source_id="pgv", source_frame_id=2, timestamp_ns=1_100_000_000)
    assert rows[0]["color"] == "GREEN"
    assert rows[0]["valid"] is True
    rows = tracker.update(red, [_detection()], source_id="pgv", source_frame_id=3, timestamp_ns=1_200_000_000)
    assert rows[0]["color"] == "UNKNOWN"
    assert rows[0]["valid"] is False
    rows = tracker.update(red, [_detection()], source_id="pgv", source_frame_id=4, timestamp_ns=1_300_000_000)
    assert rows[0]["color"] == "RED"
    assert rows[0]["valid"] is True


def test_color_evidence_expires_even_if_track_is_held():
    tracker = TrafficLightTracker(ttl_ms=1200, color_ttl_ms=500)
    frame = _lamp_frame(60)
    tracker.update(frame, [_detection()], source_id="pgv", source_frame_id=1, timestamp_ns=1_000_000_000)
    rows = tracker.update(frame, [_detection()], source_id="pgv", source_frame_id=2, timestamp_ns=1_100_000_000)
    assert rows[0]["valid"] is True
    rows = tracker.update(frame, [], source_id="pgv", source_frame_id=3, timestamp_ns=1_700_000_000)
    assert len(rows) == 1
    assert rows[0]["color"] == "GREEN"
    assert rows[0]["valid"] is False
    assert rows[0]["color_age_ms"] == 600.0
