from __future__ import annotations

import cv2
import numpy as np

from tools.adas_hmi_sim.backend.lane_runtime import FrontLaneCandidate


def _road_with_lanes() -> np.ndarray:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[180:, :] = (45, 45, 45)
    # A simple trapezoid in the forward-camera ROI.  The left boundary has a
    # negative image slope and the right boundary a positive one.
    cv2.line(image, (80, 350), (270, 190), (235, 235, 235), 8)
    cv2.line(image, (560, 350), (370, 190), (235, 235, 235), 8)
    return image


def test_cpu_lane_candidate_reports_both_boundaries() -> None:
    estimate = FrontLaneCandidate().update(_road_with_lanes())
    assert estimate.coverage == 1.0
    assert set(estimate.lanes) == {"left", "right"}
    assert all(len(item["points"]) == 5 for item in estimate.lanes.values())
    assert estimate.as_metadata()["validation"] == "EXPERIMENTAL"


def test_cpu_lane_candidate_drops_blank_frame() -> None:
    estimate = FrontLaneCandidate().update(np.zeros((360, 640, 3), dtype=np.uint8))
    assert estimate.coverage == 0.0
    assert estimate.lanes == {}


def test_cpu_lane_candidate_can_reuse_last_result_between_board_updates() -> None:
    runtime = FrontLaneCandidate(update_stride=3)
    first = runtime.update(_road_with_lanes())
    reused = runtime.update(np.zeros((360, 640, 3), dtype=np.uint8))
    reused_again = runtime.update(np.zeros((360, 640, 3), dtype=np.uint8))
    assert first.coverage == 1.0
    assert reused.coverage == 1.0
    assert reused_again.coverage == 1.0


def test_cpu_lane_candidate_keeps_short_gap_without_injecting_new_geometry() -> None:
    runtime = FrontLaneCandidate(max_missed=2)
    assert runtime.update(_road_with_lanes()).coverage == 1.0
    assert runtime.update(np.zeros((360, 640, 3), dtype=np.uint8)).coverage == 1.0
    assert runtime.update(np.zeros((360, 640, 3), dtype=np.uint8)).coverage == 1.0
    assert runtime.update(np.zeros((360, 640, 3), dtype=np.uint8)).coverage == 0.0
