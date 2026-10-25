"""Clamp YOLO split outputs in ONNX so INT8 ranges are not dominated by outliers."""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--box-min", type=float, default=-32.0)
    ap.add_argument("--box-max", type=float, default=544.0)
    ap.add_argument("--score-min", type=float, default=-12.0)
    ap.add_argument("--score-max", type=float, default=6.0)
    args = ap.parse_args()
    model = onnx.load(args.input)
    graph = model.graph
    assert [o.name for o in graph.output] == ["boxes", "scores"], [o.name for o in graph.output]

    def clip(name: str, lower: float, upper: float) -> None:
        raw = f"{name}_raw"
        producer = next(node for node in graph.node if name in node.output)
        for index, output in enumerate(producer.output):
            if output == name:
                producer.output[index] = raw
        low = helper.make_tensor(f"{name}_min", TensorProto.FLOAT, [], [lower])
        high = helper.make_tensor(f"{name}_max", TensorProto.FLOAT, [], [upper])
        graph.initializer.extend([low, high])
        graph.node.append(helper.make_node("Clip", [raw, f"{name}_min", f"{name}_max"], [name], name=f"clip_{name}"))

    clip("boxes", args.box_min, args.box_max)
    clip("scores", args.score_min, args.score_max)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print("saved", args.output)


if __name__ == "__main__":
    main()
