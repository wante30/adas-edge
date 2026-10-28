"""Expose YOLO box and class-score branches as separate ONNX outputs for RKNN."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper, shape_inference


def _shape(value_info: onnx.ValueInfoProto) -> list[int]:
    dimensions = value_info.type.tensor_type.shape.dim
    if any(not dimension.HasField("dim_value") for dimension in dimensions):
        raise ValueError(f"output {value_info.name!r} must have a static shape")
    return [dimension.dim_value for dimension in dimensions]


def _concat_axis(node: onnx.NodeProto) -> int:
    for attribute in node.attribute:
        if attribute.name == "axis":
            return attribute.i
    return 1


def split_combined_detection_output(model: onnx.ModelProto, scores_mode: str = "prob") -> onnx.ModelProto:
    """Return a copy whose final [boxes, scores] concat is exposed separately.

    ``scores_mode="prob"`` (default) exposes the post-sigmoid class probabilities,
    exactly like the deployed 7-class models. ``scores_mode="logits"`` cuts one
    node earlier and exposes the pre-sigmoid logits: INT8 quantization then works
    on a log-odds range, which preserves small probabilities that a probability-
    space quantizer flattens into its noise floor (measured on the 1-class
    traffic-light specialist: recall at conf 0.05 recovered from 0.098 to FP32
    parity by moving the cut to logits).
    """
    model = shape_inference.infer_shapes(deepcopy(model))
    graph = model.graph
    if len(graph.output) != 1:
        raise ValueError("expected exactly one combined YOLO output")

    combined_name = graph.output[0].name
    producer = next((node for node in graph.node if combined_name in node.output), None)
    if producer is None or producer.op_type != "Concat" or _concat_axis(producer) != 1:
        raise ValueError("expected the combined YOLO output to be produced by axis-1 Concat")
    if len(producer.input) != 2:
        raise ValueError("expected exactly two Concat inputs: boxes and scores")

    value_infos = {value.name: value for value in (*graph.input, *graph.value_info, *graph.output)}
    try:
        boxes_info, scores_info = (value_infos[name] for name in producer.input)
    except KeyError as error:
        raise ValueError(f"missing shape information for {error.args[0]!r}") from error

    if scores_mode == "logits":
        sigmoid = next((node for node in graph.node if scores_info.name in node.output), None)
        if sigmoid is None or sigmoid.op_type != "Sigmoid":
            raise ValueError(
                f"expected the scores tensor {scores_info.name!r} to be produced by Sigmoid, "
                f"got {sigmoid.op_type if sigmoid else 'nothing'}"
            )
        logits_name = sigmoid.input[0]
        scores_info = value_infos.get(logits_name)
        if scores_info is None:
            raise ValueError(f"missing shape information for the logits tensor {logits_name!r}")

    boxes_shape, scores_shape = _shape(boxes_info), _shape(scores_info)
    if len(boxes_shape) != 3 or boxes_shape[1] != 4:
        raise ValueError(f"expected boxes shape [batch, 4, anchors], got {boxes_shape}")
    if len(scores_shape) != 3 or scores_shape[1] < 1 or boxes_shape[0] != scores_shape[0] or boxes_shape[2] != scores_shape[2]:
        raise ValueError(f"unexpected scores shape {scores_shape} for boxes shape {boxes_shape}")

    scores_source = producer.input[1] if scores_mode == "prob" else sigmoid.input[0]
    graph.node.extend(
        [
            helper.make_node("Identity", [producer.input[0]], ["boxes"], name="rknn_boxes_output"),
            helper.make_node("Identity", [scores_source], ["scores"], name="rknn_scores_output"),
        ]
    )
    del graph.output[:]
    graph.output.extend(
        [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
        ]
    )
    checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Static YOLO detection ONNX model")
    parser.add_argument("--output", type=Path, required=True, help="ONNX model with boxes/scores outputs")
    parser.add_argument(
        "--scores-mode",
        choices=("prob", "logits"),
        default="prob",
        help="'prob' keeps post-sigmoid scores (7-class models); 'logits' cuts before the sigmoid",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(split_combined_detection_output(onnx.load(args.input), args.scores_mode), args.output)
    print(f"Separated YOLO ONNX outputs: {args.output}")


if __name__ == "__main__":
    main()
