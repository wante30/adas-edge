from __future__ import annotations

import cv2
import numpy as np

from tools.adas_hmi_sim.backend.traffic_light_runtime import TrafficLightTemporalFilter


def _frame(color: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[20:48, 45:75] = color
    return image


def test_temporal_filter_confirms_and_holds_green() -> None:
    detector = {"class_name": "traffic light", "confidence": 0.9, "bbox": [40, 15, 80, 55]}
    runtime = TrafficLightTemporalFilter(confirm_frames=2, hold_frames=1)
    green = _frame((0, 220, 0))
    assert runtime.update(green, [detector]).state == "UNKNOWN"
    assert runtime.update(green, [detector]).state == "GREEN"
    assert runtime.update(green, []).state == "GREEN"
    assert runtime.update(green, []).state == "UNKNOWN"


def test_temporal_filter_handles_red_and_unknown_pixels() -> None:
    detector = {"class_name": "traffic light", "confidence": 0.9, "bbox": [40, 15, 80, 55]}
    runtime = TrafficLightTemporalFilter(confirm_frames=1)
    assert runtime.update(_frame((0, 0, 230)), [detector]).state == "RED"
    blank = np.zeros((80, 120, 3), dtype=np.uint8)
    assert runtime.update(blank, [detector]).raw_state == "UNKNOWN"


def test_unassociated_green_cannot_authorize_ego_go() -> None:
    from tools.adas_hmi_sim.backend.scene_builder import SceneBuilder

    detector = {"class_name": "traffic light", "confidence": 0.9, "bbox": [40, 15, 80, 55]}
    estimate = TrafficLightTemporalFilter(confirm_frames=1).update(_frame((0, 220, 0)), [detector])
    assert estimate.state == "GREEN"
    scene = SceneBuilder().build(1, 100, [detector], 120, 80, traffic_light_state=estimate.state, traffic_light_decision_state=estimate.decision_state, source="front")
    assert scene.ego.action == "STOP"
    assert "TRAFFIC_LIGHT_UNKNOWN" in scene.warnings
    assert estimate.as_metadata()["association"] == "UNVERIFIED"


def test_low_confidence_colored_candidate_is_unknown_by_default() -> None:
    # Low-confidence detections cannot establish a stable color candidate.
    detector = {"class_name": "traffic light", "confidence": 0.20, "bbox": [40, 15, 80, 55]}
    estimate = TrafficLightTemporalFilter(confirm_frames=1).update(_frame((0, 0, 230)), [detector])
    assert estimate.state == "UNKNOWN"


def test_side_signal_candidate_is_not_used_for_ego_state() -> None:
    detector = {"class_name": "traffic light", "confidence": 0.9, "bbox": [90, 15, 119, 55]}
    estimate = TrafficLightTemporalFilter(confirm_frames=1).update(_frame((0, 0, 230)), [detector])
    assert estimate.state == "UNKNOWN"


def test_signal_shape_metadata_distinguishes_vehicle_and_pedestrian_lights() -> None:
    runtime = TrafficLightTemporalFilter(confirm_frames=1)
    vehicle = {"class_name": "traffic light", "confidence": 0.9, "bbox": [80, 20, 140, 50]}
    pedestrian = {"class_name": "traffic light", "confidence": 0.95, "bbox": [40, 10, 70, 90]}
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[24:44, 85:135] = (0, 220, 0)
    image[15:85, 45:65] = (0, 0, 230)
    estimate = runtime.update(image, [vehicle, pedestrian])
    assert estimate.state == "GREEN"
    assert estimate.signal_type == "VEHICLE_OR_HORIZONTAL"
    assert estimate.as_metadata()["association"] == "UNVERIFIED"


def test_pedestrian_state_has_shorter_gap_hold_than_vehicle_state() -> None:
    detector = {"class_name": "traffic light", "confidence": 0.9, "bbox": [40, 10, 70, 90]}
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[15:85, 45:65] = (0, 0, 230)
    runtime = TrafficLightTemporalFilter(confirm_frames=1, hold_frames=12)
    assert runtime.update(image, [detector]).pedestrian_state == "RED"
    for _ in range(4):
        assert runtime.update(image, []).pedestrian_state == "RED"
    assert runtime.update(image, []).pedestrian_state == "UNKNOWN"


def test_geometry_association_can_authorize_only_a_central_vehicle_signal() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[18:32, 58:102] = (0, 220, 0)
    detector = {"class_name": "traffic light", "confidence": 0.9, "bbox": [55, 14, 105, 36]}
    lanes = {
        "left": {"points": [{"x": -2.0, "z": 48.0}]},
        "right": {"points": [{"x": 2.0, "z": 48.0}]},
    }
    runtime = TrafficLightTemporalFilter(confirm_frames=1, association_mode="geometry-experimental")
    estimate = runtime.update(image, [detector], lane_metadata=lanes)
    assert estimate.state == "GREEN"
    assert estimate.decision_state == "GREEN"
    assert estimate.association == "GEOMETRIC_EXPERIMENTAL"


def test_geometry_association_rejects_side_vehicle_signal() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[18:32, 125:155] = (0, 220, 0)
    detector = {"class_name": "traffic light", "confidence": 0.9, "bbox": [122, 14, 158, 36]}
    lanes = {
        "left": {"points": [{"x": -2.0, "z": 48.0}]},
        "right": {"points": [{"x": 2.0, "z": 48.0}]},
    }
    runtime = TrafficLightTemporalFilter(confirm_frames=1, association_mode="geometry-experimental")
    estimate = runtime.update(image, [detector], lane_metadata=lanes)
    assert estimate.state == "UNKNOWN"
    assert estimate.decision_state == "UNKNOWN"
    assert estimate.association == "UNVERIFIED"
