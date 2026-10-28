"""Deterministic perception-only rendering and unchanged crop geometry.

The saved simulation UI predates removal of driving controls, so it is only
used as a reference for the unchanged source-crop geometry.
"""
import json

import cv2
import numpy as np
import pytest

from competition_ai.bound_signal import SignalObservation
from competition_ai.hazard_events import HazardObservation
from competition_ai.scenario import Controls, Direction, Light, Mode, Motion
from competition_ai.scenario_policy import decide
from tools.benchmark_cockpit_render import REFERENCE, ROOT, load_module
from tools.competition_console.pipeline import FrontSnapshot
from tools.competition_console.split_video import SplitVideo
from tools.competition_console.view import NativeView


@pytest.fixture(scope="module")
def originals():
    if not all((REFERENCE / name).is_file() for name in ("view.py", "split_video.py")):
        pytest.skip("Original A/B source snapshots are local performance evidence, not shipped dependencies")
    return (load_module(REFERENCE / "view.py", "test_reference_cockpit_view"),
            load_module(REFERENCE / "split_video.py", "test_reference_cockpit_split"))


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    old = cv2.getNumThreads()
    cv2.setNumThreads(1)
    yield
    cv2.setNumThreads(old)


@pytest.mark.parametrize("polygon", [
    [[0, 0], [99, 0], [99, 79], [0, 79]],
    [[-20, 30], [45, -30], [140, 55], [45, 110]],
    [[10, 45], [50, 47], [70, 79], [3, 78]],
    [[-40, -40], [-30, -40], [-30, -30], [-40, -30]],
    [[100, 80], [130, 80], [130, 100], [100, 100]],
])
def test_direct_bgr_fill_matches_legacy_mask_indexing_even_at_clipped_edges(polygon):
    source = np.random.default_rng(181).integers(0, 256, (80, 100, 3), dtype=np.uint8)
    polygon = np.asarray(polygon, np.int32)
    mask = np.zeros(source.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    expected = source.copy()
    expected[mask != 0] = 0
    actual = source.copy()
    cv2.fillConvexPoly(actual, polygon, (0, 0, 0))
    assert np.array_equal(actual, expected)


def test_cached_corner_patches_cover_exactly_the_full_rounded_mask_exterior():
    view = NativeView()
    expected = 255 - np.asarray(view._video_mask)
    actual = np.zeros_like(expected)
    vx, vy, _, _ = view.content_rect
    for (x, y), _, outside in view._video_corners:
        patch = np.asarray(outside)
        actual[y-vy:y-vy+patch.shape[0], x-vx:x-vx+patch.shape[1]] = patch
    assert np.array_equal(actual, expected)


def test_real_geometry_and_all_metadata_match_at_keys_and_intermediate_frames(originals):
    profile = json.loads((ROOT / "configs/competition_native_navigation_112739.json").read_text(encoding="utf-8"))
    reference = originals[1].SplitVideo(profile)
    candidate = SplitVideo(profile)
    source = np.random.default_rng(417).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    unmodified = source.copy()
    ids = candidate.ids + [(a+b)//2 for a, b in zip(candidate.ids, candidate.ids[1:])]
    for fid in ids:
        old, new = reference.split(source, fid), candidate.split(source, fid)
        assert np.array_equal(old.road, new.road), fid
        assert np.array_equal(old.navigation, new.navigation), fid
        assert old.metadata == new.metadata, fid
    assert np.array_equal(source, unmodified)


@pytest.mark.parametrize("mode,variant", [
    (Mode.INTERSECTION, "base"), (Mode.ROAD, "base"), (Mode.NAVIGATION, "base"),
    (Mode.HAZARD, "base"), (Mode.INTERSECTION, "alarm"), (Mode.NAVIGATION, "diagnostics"),
    (Mode.NAVIGATION, "unreadable"), (Mode.NAVIGATION, "ended"), (Mode.NAVIGATION, "no_overlay"),
    (Mode.INTERSECTION, "binding"), (Mode.INTERSECTION, "paused"), (Mode.INTERSECTION, "bev_geometry"),
])
def test_native_render_is_repeatable_and_keeps_input_immutable_across_states(mode, variant):
    candidate = NativeView()
    source = np.random.default_rng(252).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    nav_image = np.random.default_rng(253).integers(0, 256, (540, 960, 3), dtype=np.uint8)
    source_before, nav_before = source.copy(), nav_image.copy()
    control = Controls(mode=mode, signalized=True, navigation=Direction.RIGHT,
                       navigation_valid=True, motion=Motion.GO if variant == "alarm" else Motion.AUTO)
    signal = SignalObservation(light=Light.RED, raw_light=Light.RED, associated=True,
                               bbox=(20, 30, 130, 65), timestamp_ns=1_000_000_000)
    hazards = HazardObservation(events=({"kind": "LEAD_BRAKE_LAMP_RISE", "bbox": [240, 220, 310, 300]},))
    snapshot = FrontSnapshot([{"class_name": "car", "confidence": .9, "bbox": [200, 200, 350, 350]}],
                             [], signal, 0., 0., "SOFTWARE_TEST", hazards)
    perception = snapshot.as_perception()
    decision = decide(control, perception, 1_000_000_000)
    navigation = {"direction": "right", "distance_m": 893., "road_name": "",
                  "valid": variant != "unreadable", "reason": "software_test",
                  "source_frame_id": 30, "timestamp_ns": 1_000_000_000}
    options = {"frame_id": 30, "fps": 30, "total_frames": 300,
               "navigation": navigation, "navigation_image": nav_image,
               "diagnostics": variant == "diagnostics", "ended": variant == "ended",
               "overlay": variant != "no_overlay", "paused": variant == "paused",
               "binding": variant == "binding", "drag_box": (40, 60, 280, 100),
               "source_name": "software_pixel_equivalence_fixture"}
    if variant == "bev_geometry":
        # Only exercise the rounded compositing path; no upstream perception runs.
        candidate._bev = lambda items, shape: source.copy()
        options["bev"] = True
    mask_before = candidate._video_mask.tobytes()
    clear_before = candidate._video_clear.tobytes()
    background_before = candidate._background.tobytes()
    first = candidate.render(source, snapshot, control, decision, **options)
    first_buttons = list(candidate.buttons)
    first_rect = candidate.content_rect
    second = candidate.render(source, snapshot, control, decision, **options)
    assert np.array_equal(first, second)
    assert first.shape == (900, 1600, 3) and first.dtype == np.uint8
    assert candidate.buttons == first_buttons
    assert candidate.content_rect == first_rect
    assert {key for key,_ in candidate.buttons} <= {'o','f','c',' ','.','r','v'}
    assert np.array_equal(source, source_before)
    assert np.array_equal(nav_image, nav_before)
    assert candidate._video_mask.tobytes() == mask_before
    assert candidate._video_clear.tobytes() == clear_before
    assert candidate._background.tobytes() == background_before


def test_rgb_render_output_matches_bgr_render_with_channels_reversed():
    view = NativeView()
    source = np.random.default_rng(901).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    controls = Controls(mode=Mode.INTERSECTION)
    snapshot = FrontSnapshot([], [], SignalObservation(), None, None, "UNBOUND")
    decision = decide(controls, snapshot.as_perception(), 1_000_000_000)
    options = {"frame_id": 30, "fps": 30, "source_name": "rgb_output_test"}

    bgr = view.render(source, snapshot, controls, decision, **options)
    rgb = view.render(source, snapshot, controls, decision, output_rgb=True, **options)

    assert np.array_equal(rgb, bgr[:, :, ::-1])
