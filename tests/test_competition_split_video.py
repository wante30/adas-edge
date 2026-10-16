from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.competition_console.split_video import SplitVideo


def profile():
    return {"schema": "competition.video_split.v1", "source_size": [120, 100],
            "source_frame_count": 11, "navigation_size": [960, 540], "keyframes": [
                {"frame_id": 0, "road_rect": [0, 0, 120, 20],
                 "navigation_quad": [[10, 50], [100, 50], [100, 95], [10, 95]]},
                {"frame_id": 10, "road_rect": [0, 0, 120, 25],
                 "navigation_quad": [[20, 55], [110, 55], [110, 99], [20, 99]]}]}


def test_two_outputs_come_from_current_source_pixels_and_id():
    splitter = SplitVideo(profile())
    frame = np.full((100, 120, 3), (17, 33, 99), np.uint8)
    result = splitter.split(frame, 5)
    assert result.metadata["source_frame_id"] == result.metadata["road_source_frame_id"] == result.metadata["navigation_source_frame_id"] == 5
    assert result.road.shape == (22, 120, 3)
    assert result.navigation.shape == (540, 960, 3)
    assert np.all(result.road == [17, 33, 99])
    assert np.all(result.navigation == [17, 33, 99])
    frame[:] = (72, 81, 91)
    next_result = splitter.split(frame, 6)
    assert np.all(next_result.road == [72, 81, 91])
    assert np.all(next_result.navigation == [72, 81, 91])
    assert np.all(result.road == [17, 33, 99])  # earlier outputs are immutable snapshots of input pixels


def test_homography_maps_quad_to_output_edges():
    frame = np.zeros((100, 120, 3), np.uint8)
    splitter = SplitVideo(profile())
    result = splitter.split(frame, 0)
    source = np.asarray(result.metadata["navigation_quad"], np.float32).reshape(1, 4, 2)
    matrix = np.asarray(result.metadata["navigation_transform"])
    actual = cv2.perspectiveTransform(source, matrix)[0]
    assert np.allclose(actual, [[0, 0], [959, 0], [959, 539], [0, 539]], atol=.001)


def test_partial_screen_is_black_padding_with_coverage_metadata():
    config = profile()
    for key in config["keyframes"]:
        key["navigation_quad"] = [[-10, 50], [100, 50], [100, 120], [-10, 120]]
    result = SplitVideo(config).split(np.full((100, 120, 3), 180, np.uint8), 2)
    assert result.metadata["navigation_partially_outside_source"]
    assert .5 < result.metadata["navigation_visible_fraction"] < .8
    assert np.all(result.navigation[-1] == 0)
    assert result.metadata["unobserved_navigation_pixels"] == "BLACK_PADDING_NOT_RECONSTRUCTED"


@pytest.mark.parametrize("fid", [-1, 11, 1.2, True])
def test_rejects_frame_ids_outside_exact_source(fid):
    with pytest.raises(ValueError):
        SplitVideo(profile()).split(np.zeros((100, 120, 3), np.uint8), fid)


@pytest.mark.parametrize("shape,dtype", [((99, 120, 3), np.uint8), ((100, 120), np.uint8), ((100, 120, 3), np.float32)])
def test_rejects_wrong_source_shape_or_type(shape, dtype):
    with pytest.raises(ValueError):
        SplitVideo(profile()).split(np.zeros(shape, dtype), 0)


@pytest.mark.parametrize("mutation", ["bounds", "bad_mask", "bowtie", "unordered_keys", "incomplete"])
def test_rejects_invalid_geometric_calibration(mutation):
    config = deepcopy(profile())
    if mutation == "bounds": config["keyframes"][0]["road_rect"] = [-1, 0, 120, 20]
    elif mutation == "bad_mask": config["road_screen_mask_margin"] = [-1, 0, 0, 0]
    elif mutation == "bowtie": config["keyframes"][0]["navigation_quad"] = [[10, 50], [100, 95], [100, 50], [10, 95]]
    elif mutation == "unordered_keys": config["keyframes"].reverse()
    else: config["keyframes"][-1]["frame_id"] = 9
    with pytest.raises(ValueError):
        SplitVideo(config)


def test_real_profile_contains_only_geometry_and_covers_all_frames():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/competition_native_navigation_112739.json").read_text(encoding="utf-8"))
    geometry = config["split_geometry"]
    assert set(geometry) == {"schema", "source_size", "source_frame_count", "navigation_size", "keyframes", "road_screen_mask_margin"}
    for key in geometry["keyframes"]:
        assert set(key) == {"frame_id", "road_rect", "navigation_quad"}
    splitter = SplitVideo(config)
    # Full source coverage catches non-half interpolation weights and the
    # previously observed width 1920.0000000000002 rounding regression.
    frame = np.zeros((1080, 1920, 3), np.uint8)
    for fid in range(geometry["source_frame_count"]):
        result = splitter.split(frame, fid)
        assert result.road.size and result.navigation.shape == (540, 960, 3)
        assert 0 <= result.metadata["road_masked_fraction"] < 1


def test_screen_is_removed_from_road_without_modifying_navigation_or_source():
    config = profile()
    config["road_screen_mask_margin"] = [2, 3, 2, 4]
    for key in config["keyframes"]:
        key["road_rect"] = [0, 0, 120, 100]
    frame = np.full((100, 120, 3), (33, 66, 99), np.uint8)
    result = SplitVideo(config).split(frame, 0)
    assert np.all(result.road[48:99, 8:103] == 0)
    assert np.all(result.road[:45] == [33, 66, 99])
    assert np.all(result.navigation == [33, 66, 99])
    assert np.all(frame == [33, 66, 99])
    assert result.metadata["road_masked_fraction"] > 0
