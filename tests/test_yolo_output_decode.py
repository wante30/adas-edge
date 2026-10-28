"""Tests for decoding split YOLO outputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_decoder_module():
    path = Path(__file__).parents[1] / "deployment" / "rknn" / "decode_yolo_output.py"
    spec = importlib.util.spec_from_file_location("decode_yolo_output", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_split_outputs_returns_xyxy_detection() -> None:
    decoder = _load_decoder_module()
    boxes = np.zeros((1, 4, 3), dtype=np.float32)
    boxes[0, :, 1] = [320, 320, 100, 80]
    scores = np.zeros((1, 80, 3), dtype=np.float32)
    scores[0, 2, 1] = 0.9

    detections = decoder.decode_outputs([boxes, scores], confidence_threshold=0.25, iou_threshold=0.5)

    assert len(detections) == 1
    assert detections[0]["class_id"] == 2
    assert detections[0]["confidence"] == pytest.approx(0.9)
    assert detections[0]["bbox"] == [270.0, 280.0, 370.0, 360.0]


def test_rectangular_raw_head_uses_independent_xy_strides() -> None:
    profiler = __import__("tools.profile_rknn_single_stream", fromlist=["decode_yolo11_raw"])
    outputs = []
    for grid_h, grid_w in ((18, 32), (9, 16), (4, 8)):
        reg = np.full((1, 64, grid_h, grid_w), -10.0, dtype=np.float32)
        cls = np.full((1, 80, grid_h, grid_w), -10.0, dtype=np.float32)
        reg[:, 0::16, grid_h // 2, grid_w // 2] = 1.0
        cls[:, 2, grid_h // 2, grid_w // 2] = 10.0
        outputs.extend((reg, cls))
    boxes = profiler.decode_yolo11_raw(outputs, (512, 288), 0.25, 0.45, 1.0, 0, 0)
    assert boxes
    assert all(0 <= row[0] <= 512 and 0 <= row[2] <= 512 for row in boxes)
    assert all(0 <= row[1] <= 288 and 0 <= row[3] <= 288 for row in boxes)
