"""Expose per-scale YOLO box/score branches as separate ONNX outputs (N scales)."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper, shape_inference


def shape_of(value_info: onnx.ValueInfoProto) -> list[int]:
    dims = value_info.type.tensor_type.shape.dim
    if any(not d.HasField("dim_value") for d in dims):
        raise ValueError(f"dynamic shape on {value_info.name}")
    return [d.dim_value for d in dims]


def concat_axis(node: onnx.NodeProto) -> int:
    for attribute in node.attribute:
        if attribute.name == "axis":
            return attribute.i
    return 1


def split(model: onnx.ModelProto, scores_mode: str) -> onnx.ModelProto:
    model = shape_inference.infer_shapes(deepcopy(model))
    graph = model.graph
    if len(graph.output) != 1:
        raise ValueError("expected one combined output")
    combined = graph.output[0].name
    producer = next(node for node in graph.node if combined in node.output)
    if producer.op_type != "Concat" or concat_axis(producer) != 1:
        raise ValueError("combined output is not an axis-1 Concat")

    values = {v.name: v for v in (*graph.input, *graph.value_info, *graph.output)}
    outputs = []
    new_nodes = []
    for scale, tensor_name in enumerate(producer.input):
        scale_node = next((node for node in graph.node if tensor_name in node.output), None)
        if scale_node is None or scale_node.op_type != "Concat" or len(scale_node.input) != 2:
            raise ValueError(f"scale {scale} is not a two-input Concat")
        boxes_name, scores_name = scale_node.input
        boxes_shape = shape_of(values[boxes_name])
        scores_source = scores_name
        if scores_mode == "logits":
            sigmoid = next(node for node in graph.node if scores_name in node.output)
            if sigmoid.op_type != "Sigmoid":
                raise ValueError(f"scale {scale} scores are not produced by Sigmoid")
            scores_source = sigmoid.input[0]
        scores_shape = shape_of(values[scores_source])
        box_out, score_out = f"boxes_{scale}", f"scores_{scale}"
        new_nodes.append(helper.make_node("Identity", [boxes_name], [box_out], name=f"boxes_out_{scale}"))
        new_nodes.append(helper.make_node("Identity", [scores_source], [score_out], name=f"scores_out_{scale}"))
        outputs.append(helper.make_tensor_value_info(box_out, TensorProto.FLOAT, boxes_shape))
        outputs.append(helper.make_tensor_value_info(score_out, TensorProto.FLOAT, scores_shape))
    graph.node.extend(new_nodes)
    del graph.output[:]
    graph.output.extend(outputs)
    checker.check_model(model)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scores-mode", choices=("prob", "logits"), default="logits")
    args = ap.parse_args()
    model = split(onnx.load(args.input), args.scores_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print("saved", args.output, "outputs", [o.name for o in model.graph.output])


if __name__ == "__main__":
    main()
