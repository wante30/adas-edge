"""CLI safety checks for RKNN conversion."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_converter_module():
    path = Path(__file__).parents[1] / "deployment" / "rknn" / "convert_rknn.py"
    spec = importlib.util.spec_from_file_location("convert_rknn", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_yolo_rknn_converter_defaults_to_fp16_without_calibration() -> None:
    converter = _load_converter_module()

    args = converter.parse_args(["--onnx", "model.onnx", "--output", "model.rknn"])

    assert args.precision == "fp16"
    assert args.dataset is None
