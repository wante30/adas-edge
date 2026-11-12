import numpy as np
from types import SimpleNamespace

from competition_ai.scenario import Light
from tools.competition_console.fpga_perception import build_hud_perception


def test_maps_only_directly_observed_fields_and_leaves_unavailable_inputs_clear():
    road = np.zeros((120, 200, 3), np.uint8)
    signal = np.zeros((30, 12, 3), np.uint8)
    signal[6:16, 3:9] = (0, 0, 255)
    road[10:40, 20:32] = signal
    snapshot = type("Snapshot", (), {
        "main": [{"class_name": "person", "confidence": 0.8, "bbox": [40, 10, 70, 90]}],
        "specialist": [{"class_name": "traffic light", "confidence": 0.9,
                        "bbox": [20, 10, 32, 40]}],
    })()

    perception = build_hud_perception(road, snapshot)

    assert perception.traffic.name == "RED"
    assert perception.pedestrian is True
    assert perception.static_target is False
    assert perception.overspeed is False
    assert perception.lane_left == perception.lane_right == 0
    assert perception.lka_warning is False
    assert perception.traffic_evidence == "ONE_DETECTED_RED_SIGNAL"


def test_yellow_and_duplicate_yellow_boxes_never_become_red_or_green():
    road = np.zeros((100, 160, 3), np.uint8)
    yellow = np.zeros((30, 12, 3), np.uint8)
    yellow[6:16, 3:9] = (0, 255, 255)
    road[10:40, 10:22] = yellow
    lights = [{"class_name": "traffic light", "confidence": 0.9,
               "bbox": [10, 10, 22, 40]}]
    snapshot = type("Snapshot", (), {"main": [], "specialist": lights})()

    result = build_hud_perception(road, snapshot)

    assert result.traffic.name == "OFF"
    assert result.traffic_evidence == "YELLOW_UNSUPPORTED_BY_HUD_PROTOCOL"

    snapshot.specialist = lights * 2
    result = build_hud_perception(road, snapshot)
    assert result.traffic.name == "OFF"
    assert result.traffic_evidence == "YELLOW_UNSUPPORTED_BY_HUD_PROTOCOL"


def test_fresh_user_bound_red_signal_overrides_unassociated_detector_candidates():
    frame = np.zeros((100, 160, 3), np.uint8)
    snapshot = type("Snapshot", (), {
        "main": [],
        "specialist": [
            {"class_name": "traffic light", "confidence": 0.08, "bbox": [10, 10, 20, 40]},
            {"class_name": "traffic light", "confidence": 0.06, "bbox": [40, 10, 50, 40]},
        ],
        "signal": SimpleNamespace(
            light=Light.RED, associated=True, timestamp_ns=8_000_000_000, reason="observed"
        ),
    })()

    result = build_hud_perception(frame, snapshot)

    assert result.traffic.name == "RED"
    assert result.ext_request() == 3
    assert result.traffic_evidence == "USER_BOUND_RED_SIGNAL"


def test_confident_red_signal_survives_weak_specialist_false_positive():
    frame = np.zeros((100, 180, 3), np.uint8)
    frame[17:24, 13:19] = (0, 0, 255)
    snapshot = type("Snapshot", (), {
        "main": [],
        "specialist": [
            {"class_name": "traffic light", "confidence": 0.9, "bbox": [10, 10, 22, 40]},
            {"class_name": "traffic light", "confidence": 0.08, "bbox": [60, 10, 72, 40]},
        ],
    })()

    result = build_hud_perception(frame, snapshot)

    assert result.traffic.name == "RED"


def test_overlapping_main_and_specialist_boxes_for_one_red_signal_are_deduplicated():
    frame = np.zeros((100, 180, 3), np.uint8)
    frame[17:24, 13:19] = (0, 0, 255)
    snapshot = type("Snapshot", (), {
        "main": [{"class_name": "traffic light", "confidence": 0.82,
                  "bbox": [11, 10, 23, 40]}],
        "specialist": [
            {"class_name": "traffic light", "confidence": 0.9, "bbox": [10, 10, 22, 40]},
            {"class_name": "traffic light", "confidence": 0.78, "bbox": [11, 10, 23, 40]},
        ],
    })()

    result = build_hud_perception(frame, snapshot)

    assert result.traffic.name == "RED"
    assert result.traffic_evidence == "ONE_DETECTED_RED_SIGNAL"


def test_distinct_confident_red_and_green_signals_remain_off():
    frame = np.zeros((100, 180, 3), np.uint8)
    frame[17:24, 13:19] = (0, 0, 255)
    frame[17:24, 83:89] = (0, 255, 0)
    snapshot = type("Snapshot", (), {
        "main": [],
        "specialist": [
            {"class_name": "traffic light", "confidence": 0.9, "bbox": [10, 10, 22, 40]},
            {"class_name": "traffic light", "confidence": 0.88, "bbox": [80, 10, 92, 40]},
        ],
    })()

    result = build_hud_perception(frame, snapshot)

    assert result.traffic.name == "OFF"
    assert result.traffic_evidence == "CONFLICTING_CONFIRMED_SIGNAL_COLORS"


def test_confirmed_multi_signal_red_maps_to_traffic_without_using_arrow_as_lane_state():
    snapshot = type("Snapshot", (), {"main": [], "specialist": []})()
    multi_signal = {
        "signals": [{
            "track_id": 7,
            "valid": True,
            "color": "RED",
            "shape": "LEFT_ARROW",
            "confidence": 0.91,
        }],
        "directions": {"left": {"color": "RED", "valid": True}},
    }

    result = build_hud_perception(None, snapshot, multi_signal=multi_signal)

    assert result.traffic.name == "RED"
    assert result.traffic_evidence == "MULTI_SIGNAL_RED"
    assert result.ext_request() == 3
    assert result.lane_left.name == "NONE"
    assert result.lane_right.name == "NONE"


def test_multi_signal_green_is_mapped_but_yellow_and_unknown_are_released():
    snapshot = type("Snapshot", (), {"main": [], "specialist": []})()
    green = {"signals": [{"valid": True, "color": "GREEN", "shape": "ROUND"}]}
    yellow = {"signals": [{"valid": True, "color": "YELLOW", "shape": "ROUND"}]}
    unknown = {"signals": [{"valid": False, "color": "RED", "shape": "LEFT_ARROW"}]}

    assert build_hud_perception(None, snapshot, multi_signal=green).traffic.name == "GREEN"
    yellow_result = build_hud_perception(None, snapshot, multi_signal=yellow)
    assert yellow_result.traffic.name == "OFF"
    assert yellow_result.traffic_evidence == "MULTI_SIGNAL_YELLOW_UNSUPPORTED_BY_HUD_PROTOCOL"
    unknown_result = build_hud_perception(None, snapshot, multi_signal=unknown)
    assert unknown_result.traffic.name == "OFF"
    assert unknown_result.traffic_evidence == "MULTI_SIGNAL_NO_VALID_SIGNAL"


def test_multi_signal_conflicting_confirmed_colors_remain_off():
    snapshot = type("Snapshot", (), {"main": [], "specialist": []})()
    multi_signal = {
        "signals": [
            {"valid": True, "color": "RED", "shape": "ROUND"},
            {"valid": True, "color": "GREEN", "shape": "STRAIGHT_ARROW"},
        ]
    }

    result = build_hud_perception(None, snapshot, multi_signal=multi_signal)

    assert result.traffic.name == "OFF"
    assert result.traffic_evidence == "MULTI_SIGNAL_CONFLICTING_CONFIRMED_SIGNAL_COLORS"
