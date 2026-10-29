from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
ADAPTER_PATH = ROOT / "deployment" / "lubancat" / "qt_gpu_probe" / "b05p_competition_bridge.py"
CONFIG_PATH = ROOT / "deployment" / "lubancat" / "qt_gpu_probe" / "tools" / "zz-b05p-model-B-single.conf"
INSPECT_PATH = ROOT / "deployment" / "rknn" / "inspect_live_outputs.py"


def _load_adapter():
    spec = importlib.util.spec_from_file_location("b05p_competition_bridge_test", ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_inspector():
    spec = importlib.util.spec_from_file_location("inspect_live_outputs_test", INSPECT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_b_live_candidate_starts_on_the_current_q4_road_feed():
    command = next(
        line.removeprefix("ExecStart=")
        for line in CONFIG_PATH.read_text(encoding="utf-8").splitlines()
        if line.startswith("ExecStart=/")
    )

    assert "--road-quadrant 3" in command


def test_competition_default_b_uses_the_b05n_single_model():
    command = next(
        line.removeprefix("ExecStart=")
        for line in CONFIG_PATH.read_text(encoding="utf-8").splitlines()
        if line.startswith("ExecStart=/")
    )

    assert "--main-model /home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn" in command
    assert "bdd_yolov8n_512x288_raw_heads_int8.rknn" not in command
    assert "--specialist-model" not in command


def test_synthetic_replay_places_road_in_the_quadrant_used_for_inference(tmp_path):
    adapter = _load_adapter()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = np.full((36, 64, 3), (20, 90, 220), dtype=np.uint8)
    assert cv2.imwrite(str(image_dir / "road.png"), image)

    frames = adapter._make_replay_frames(tmp_path, tmp_path / "replay", 1)
    raw = frames[0].read_bytes()
    mosaic = np.frombuffer(raw, dtype=np.uint8).reshape(720, 1280, 2)
    quadrants = {
        index: cv2.cvtColor(
            mosaic[(index // 2) * 360:(index // 2 + 1) * 360,
                   (index % 2) * 640:(index % 2 + 1) * 640],
            cv2.COLOR_BGR5652BGR,
        )
        for index in range(4)
    }

    expected_quadrant = 3
    assert adapter.REPLAY_ROAD_QUADRANT == expected_quadrant
    assert adapter._replay_quadrant_args() == ["--road-quadrant", str(expected_quadrant)]
    assert int(quadrants[expected_quadrant].mean()) > 0
    assert all(int(quadrants[index].mean()) == 0 for index in range(4) if index != expected_quadrant)


def test_live_output_inspection_requires_an_explicit_quadrant():
    parser = _load_inspector().build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "model.rknn", "--frame-snapshot", "frame.mfr1"])

    args = parser.parse_args([
        "--model", "model.rknn", "--frame-snapshot", "frame.mfr1", "--quadrant", "3"
    ])
    assert args.quadrant == 3
