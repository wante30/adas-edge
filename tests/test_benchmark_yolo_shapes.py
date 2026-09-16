"""Behavior checks for the shape-sweep latency benchmark helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_tool():
    path = Path(__file__).parents[1] / "tools" / "benchmark_yolo_shapes.py"
    spec = importlib.util.spec_from_file_location("benchmark_yolo_shapes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_shape_maps_ultralytics_height_width_order() -> None:
    tool = _load_tool()

    assert tool.parse_shape("320") == 320
    # Ultralytics takes (height, width) for a rectangular imgsz.
    assert tool.parse_shape("512x288") == (288, 512)
    assert tool.parse_shape("512×288") == (288, 512)
    assert tool.parse_shape("512*288") == (288, 512)
    assert tool.parse_shape("512X288") == (288, 512)


def test_describe_shape_reports_width_first() -> None:
    tool = _load_tool()

    assert tool.describe_shape(320) == "320x320"
    assert tool.describe_shape((288, 512)) == "512x288"


def test_parse_shape_rejects_garbage() -> None:
    tool = _load_tool()

    for invalid in ("", "abc", "512x", "x288"):
        with pytest.raises((ValueError, IndexError)):
            tool.parse_shape(invalid)
