"""Behavior checks for the rectangular model build manifest collector."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]


def _load_tool():
    path = REPO / "tools" / "rect_model_manifest.py"
    spec = importlib.util.spec_from_file_location("rect_model_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sha256_file_matches_hashlib_and_handles_missing(tmp_path: Path) -> None:
    tool = _load_tool()
    payload = b"yolo11n-static-canvas"
    path = tmp_path / "weights.bin"
    path.write_bytes(payload)

    assert tool.sha256_file(path) == hashlib.sha256(payload).hexdigest().upper()
    assert tool.sha256_file(tmp_path / "absent.bin") is None


def _tiny_onnx(path: Path, shape: list[int]) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output0"])],
        "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, 3024])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    onnx.save(model, str(path))


def test_onnx_io_shapes_reads_a_static_graph(tmp_path: Path) -> None:
    tool = _load_tool()
    path = tmp_path / "m.onnx"
    _tiny_onnx(path, [1, 3, 288, 512])

    io = tool.onnx_io_shapes(path)

    assert io is not None
    assert io["inputs"][0]["shape"] == [1, 3, 288, 512]
    assert io["inputs"][0]["dynamic"] is False
    assert io["outputs"][0]["shape"] == [1, 84, 3024]
    assert tool.onnx_io_shapes(tmp_path / "absent.onnx") is None


def test_collect_records_missing_artifacts_without_raising(tmp_path: Path) -> None:
    tool = _load_tool()
    weights = tmp_path / "yolo11n.pt"
    weights.write_bytes(b"checkpoint")
    models = tmp_path / "models"
    models.mkdir()

    record = tool.collect(models, weights, [(320, 192), (512, 288)])

    assert [entry["canvas"] for entry in record["canvases"]] == ["320x192", "512x288"]
    assert record["canvases"][0]["canvas_pixels"] == 320 * 192
    assert record["canvases"][0]["rknn_int8"]["present"] is False
    assert record["canvases"][0]["static_onnx"]["sha256"] is None
    assert record["weights"]["sha256"] is not None
    assert record["quant_config"]["target_platform"] == "rk3566"


def test_collect_reads_calibration_provenance(tmp_path: Path) -> None:
    tool = _load_tool()
    weights = tmp_path / "yolo11n.pt"
    weights.write_bytes(b"checkpoint")
    models = tmp_path / "models"
    models.mkdir()
    (models / "calib_512x288.json").write_text(
        json.dumps(
            {
                "canvas": "512x288",
                "manifest": "/mnt/d/tmp/calib_512x288.txt",
                "samples": 76,
                "sources": [{"source": "traffic.mp4", "frames": 66}],
                "channel_order": "RGB",
                "layout": "NCHW",
                "array_shape": [1, 3, 288, 512],
                "frozen_holdout_used": False,
            }
        ),
        encoding="utf-8",
    )

    record = tool.collect(models, weights, [(512, 288)])
    calibration = record["canvases"][0]["calibration"]

    assert calibration["samples"] == 76
    assert calibration["layout"] == "NCHW"
    assert calibration["frozen_holdout_used"] is False


def test_render_markdown_flags_missing_rknn_models(tmp_path: Path) -> None:
    tool = _load_tool()
    weights = tmp_path / "yolo11n.pt"
    weights.write_bytes(b"checkpoint")
    models = tmp_path / "models"
    models.mkdir()

    record = tool.collect(models, weights, [(320, 192), (640, 352)])
    table = tool.render_markdown(record)

    assert "320x192" in table
    assert "640x352" in table
    assert "RKNN models missing: ['320x192', '640x352']" in table
    assert "rknn-toolkit2 2.3.2" in table
