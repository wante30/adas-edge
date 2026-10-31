"""B-05L ONNX structure audit for RK3566 deployment feasibility.

This does not claim an RKNN conversion.  It checks the fixed-shape ONNX graph,
operator inventory, input/output contract and ONNX checker status; RKNN Toolkit2
was not available in a working Python environment on this machine.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(str(args.model))
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model)
    ops = Counter(node.op_type for node in model.graph.node)
    initializers = sum(1 for _ in model.graph.initializer)
    result = {
        "model": str(args.model.resolve()),
        "size_bytes": args.model.stat().st_size,
        "ir_version": model.ir_version,
        "opset": [{"domain": item.domain or "ai.onnx", "version": item.version} for item in model.opset_import],
        "checker": "passed",
        "inputs": [{"name": value.name, "shape": [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim],
                    "dtype": value.type.tensor_type.elem_type} for value in model.graph.input],
        "outputs": [{"name": value.name, "shape": [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim],
                     "dtype": value.type.tensor_type.elem_type} for value in model.graph.output],
        "nodes": len(model.graph.node),
        "initializers": initializers,
        "operators": dict(sorted(ops.items())),
        "dynamic_dimensions": [
            {"name": value.name, "shape": [dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                                          for dim in value.type.tensor_type.shape.dim]}
            for value in inferred.graph.input
        ],
        "rknn_conversion": "not_executed",
        "rknn_note": (
            "RKNN Toolkit2 2.3.2 exists in a WSL venv, but its ONNX package is broken "
            "(onnx_cpp2py_export has no extension). No conversion result is claimed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
