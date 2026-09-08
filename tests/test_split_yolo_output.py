"""Regression test for RKNN-friendly YOLO output separation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def _load_splitter_module():
    path = Path(__file__).parents[1] / "deployment" / "rknn" / "split_yolo_output.py"
    spec = importlib.util.spec_from_file_location("split_yolo_output", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_with_combined_detection_output() -> onnx.ModelProto:
    boxes = helper.make_tensor_value_info("raw_boxes", TensorProto.FLOAT, [1, 4, 8400])
    scores = helper.make_tensor_value_info("raw_scores", TensorProto.FLOAT, [1, 80, 8400])
    output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 84, 8400])
    graph = helper.make_graph(
        [helper.make_node("Concat", ["raw_boxes", "raw_scores"], ["output0"], axis=1)],
        "combined_detection_output",
        [boxes, scores],
        [output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])


def test_splitter_exposes_box_and_score_outputs_separately() -> None:
    splitter = _load_splitter_module()

    split_model = splitter.split_combined_detection_output(_model_with_combined_detection_output())

    assert [value.name for value in split_model.graph.output] == ["boxes", "scores"]
    assert [dim.dim_value for dim in split_model.graph.output[0].type.tensor_type.shape.dim] == [1, 4, 8400]
    assert [dim.dim_value for dim in split_model.graph.output[1].type.tensor_type.shape.dim] == [1, 80, 8400]
