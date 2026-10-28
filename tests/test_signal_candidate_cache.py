import cv2
import numpy as np

from tools.adas_hmi_sim.backend.traffic_light_specialist import SignalCandidateCache
from tools.adas_hmi_sim.backend.traffic_light_runtime import TrafficLightTemporalFilter
from competition_ai.traffic_light import classify_traffic_light


def lamp_frame():
    image = np.full((100, 200, 3), 200, np.uint8)
    image[15:45, 90:150] = (20, 20, 20)
    cv2.circle(image, (137, 30), 7, (0, 255, 0), -1)
    return image


def item(conf=.1):
    return {"class_name": "traffic light", "bbox": [90, 15, 150, 45],
            "confidence": conf, "source": "TRAFFIC_LIGHT_SPECIALIST_PC"}


def test_low_score_specialist_is_not_discarded_by_main_threshold():
    e = TrafficLightTemporalFilter(confirm_frames=1).update(lamp_frame(), [item()])
    assert e.state == "GREEN"
    assert e.decision_state == "UNKNOWN"
    main_item = {**item(), "source": "MAIN_ONNX_PC"}
    assert TrafficLightTemporalFilter(confirm_frames=1).update(lamp_frame(), [main_item]).state == "UNKNOWN"


def test_lamp_evidence_rejects_uniform_green_scene_patch():
    leaves = np.full((100, 200, 3), (0, 220, 0), np.uint8)
    assert classify_traffic_light(leaves, item(.9), require_lamp_evidence=True) == "UNKNOWN"


def test_invalid_source_box_cannot_become_green_on_later_frame():
    cache = SignalCandidateCache()
    blank = np.full((100, 200, 3), 140, np.uint8)
    cache.store(blank, [item(.9)], frame_id=0, timestamp_ms=0)
    later = cache.current(100)
    assert later[0]["lamp_evidence_valid"] is False
    assert TrafficLightTemporalFilter(confirm_frames=1).update(lamp_frame(), later).state == "UNKNOWN"


def test_cache_expires_and_resets_on_source_timestamp_rollback():
    cache = SignalCandidateCache(max_age_ms=400)
    cache.store(lamp_frame(), [item(.9)], frame_id=20, timestamp_ms=1000)
    assert cache.current(1399)
    assert cache.current(1401) == []
    cache.store(lamp_frame(), [item(.9)], frame_id=20, timestamp_ms=1000)
    assert cache.current(0) == []


def test_right_visible_signal_is_display_only_after_candidate_validation():
    frame = np.full((100, 200, 3), 200, np.uint8)
    frame[15:45, 155:195] = (20, 20, 20)
    cv2.circle(frame, (180, 30), 7, (0, 255, 0), -1)
    d = {**item(.9), "bbox": [155, 15, 195, 45]}
    cache = SignalCandidateCache(); cache.store(frame, [d], frame_id=0, timestamp_ms=0)
    e = TrafficLightTemporalFilter(confirm_frames=1, ego_x_max=.95).update(frame, cache.current(0))
    assert e.state == "GREEN"
    assert e.decision_state == "UNKNOWN"


def test_single_bulb_is_not_mislabelled_pedestrian_inside_horizontal_housing():
    frame = lamp_frame()
    bulb = {**item(), "bbox": [130, 23, 144, 37]}
    cache = SignalCandidateCache(); cache.store(frame, [bulb], frame_id=0, timestamp_ms=0)
    e = TrafficLightTemporalFilter(confirm_frames=1).update(frame, cache.current(0))
    assert e.signal_type == "VEHICLE_OR_HORIZONTAL"
    assert e.state == "GREEN"
    assert e.decision_state == "UNKNOWN"


def test_unknown_shape_specialist_patch_does_not_establish_vehicle_red():
    frame = np.zeros((100, 200, 3), np.uint8)
    cv2.circle(frame, (100, 30), 8, (0, 0, 255), -1)
    d = {**item(.8), "bbox": [85, 15, 115, 45], "lamp_evidence_valid": True,
         "signal_type_hint": None}
    e = TrafficLightTemporalFilter(confirm_frames=1).update(frame, [d])
    assert e.state == "UNKNOWN"


def test_cached_observation_does_not_reclassify_new_pixels_in_old_box():
    source = lamp_frame()
    cache = SignalCandidateCache(); cache.store(source, [item(.9)], frame_id=0, timestamp_ms=0)
    later = source.copy()
    later[15:45,90:150] = (0,0,255)
    e = TrafficLightTemporalFilter(confirm_frames=1).update(later, cache.current(100))
    assert e.state == "GREEN"
    # Once its source-time TTL expires, a new filter has no observation.
    assert TrafficLightTemporalFilter(confirm_frames=1).update(later,cache.current(401)).state == "UNKNOWN"
