from __future__ import annotations

import numpy as np
import cv2
import json

from tools.adas_hmi_sim.backend.surround_bev import SurroundBEVComposer
from tools.adas_hmi_sim.backend.video_pipeline import SurroundVideoSource
from tools.adas_hmi_sim.backend.calibrated_surround import CalibratedSurroundComposer, build_luts


def test_surround_bev_composes_all_camera_roles() -> None:
    composer = SurroundBEVComposer({"output_size": [320, 320], "feather_px": 3})
    frames = {
        "front": np.full((80, 120, 3), (20, 40, 80), dtype=np.uint8),
        "left": np.full((80, 120, 3), (30, 90, 30), dtype=np.uint8),
        "right": np.full((80, 120, 3), (90, 30, 30), dtype=np.uint8),
        "back": np.full((80, 120, 3), (70, 70, 20), dtype=np.uint8),
    }
    result = composer.compose(frames)
    assert result.shape == (320, 320, 3)
    assert result.dtype == np.uint8
    assert int(result.max()) > 0
    assert composer.metadata()["renderer"] == "cached_planar_warp"


def test_surround_bev_accepts_partial_streams_for_front_first_demo() -> None:
    composer = SurroundBEVComposer({"output_size": [256, 256], "feather_px": 3})
    result = composer.compose({"front": np.full((64, 96, 3), 120, dtype=np.uint8)})
    assert result.shape == (256, 256, 3)
    # The front image occupies a non-empty region and the car overlay is drawn.
    assert np.count_nonzero(result) > 256


def test_surround_video_source_emits_bev_and_explicit_detector_boundary(tmp_path) -> None:
    for role, color in (("front", (20, 40, 80)), ("left", (30, 90, 30)), ("right", (90, 30, 30)), ("rear", (70, 70, 20))):
        path = tmp_path / f"{role}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (96, 64))
        for _ in range(2):
            writer.write(np.full((64, 96, 3), color, dtype=np.uint8))
        writer.release()

    config_path = tmp_path / "bev.json"
    config_path.write_text(json.dumps({"output_size": [128, 128], "input_size": [96, 64], "feather_px": 3}), encoding="utf-8")
    source = SurroundVideoSource(tmp_path, config_path)
    packet = source.next()
    assert packet is not None
    assert packet.surround_bev is not None
    assert packet.metadata["system_mode"] == "SURROUND_BEV"
    assert packet.metadata["rk3566"]["detector"] == "NOT_CONNECTED"
    assert packet.detections == []
    source.close()


def test_calibrated_lut_composer_loads_upstream_style_package(tmp_path) -> None:
    calibration = {
        "intrinsic": {"K": [[40, 0, 32], [0, 40, 32], [0, 0, 1]], "D": [0, 0, 0, 0]},
        "extrinsic": {
            role: {"rvec": [0, 0, 0], "tvec": [0, 0, 3]} for role in ("front", "left", "right", "back")
        },
    }
    build_luts(calibration, tmp_path, (64, 64), (64, 64), "bowl")
    composer = CalibratedSurroundComposer(tmp_path)
    output = composer.compose({role: np.full((64, 64, 3), 100, dtype=np.uint8) for role in ("front", "left", "right", "rear")})
    assert output.shape == (64, 64, 3)
    assert composer.metadata()["renderer"] == "open3d_calibrated_cached_lut"
    composer.close()
