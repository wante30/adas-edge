"""Create a B-2a variant whose three segmentation outputs are 2x2 average-pooled in-graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


SEG = ("lane_cat", "lane_style", "drivable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model = onnx.load(args.input)
    graph = model.graph
    out_names = [o.name for o in graph.output]
    for name in SEG:
        if name not in out_names:
            raise SystemExit(f"missing output {name}: {out_names}")

    new_outputs = []
    for out in list(graph.output):
        if out.name not in SEG:
            new_outputs.append(out)
            continue
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        if len(dims) != 4 or dims[2] % 2 or dims[3] % 2:
            raise SystemExit(f"unexpected seg shape {dims}")
        new_name = f"{out.name}_256"
        node = helper.make_node(
            "AveragePool", [out.name], [new_name],
            name=f"seg_downsample_{out.name}",
            kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0],
            count_include_pad=0,
        )
        graph.node.append(node)
        new_dims = [dims[0], dims[1], dims[2] // 2, dims[3] // 2]
        new_outputs.append(helper.make_tensor_value_info(new_name, TensorProto.FLOAT, new_dims))
        print(f"{out.name} {dims} -> {new_name} {new_dims}")

    del graph.output[:]
    graph.output.extend(new_outputs)
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print("saved", args.output, Path(args.output).stat().st_size)

    import onnxruntime as ort
    rng = np.random.default_rng(0)
    x = rng.random((1, 3, 288, 512), dtype=np.float32)
    a = ort.InferenceSession(args.input, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    oa = a.run(None, {a.get_inputs()[0].name: x})
    ob = b.run(None, {b.get_inputs()[0].name: x})
    an = [o.name for o in onnx.load(args.input).graph.output]
    bn = [o.name for o in onnx.load(args.output).graph.output]
    for i, name in enumerate(an[:12]):
        assert bn[i] == name
        np.testing.assert_allclose(oa[i], ob[i], rtol=1e-6, atol=1e-6, err_msg=name)
    for j, name in enumerate(SEG):
        src = oa[an.index(name)]
        pooled = src.reshape(src.shape[0], src.shape[1], src.shape[2] // 2, 2, src.shape[3] // 2, 2).mean((3, 5))
        dst = ob[bn.index(f"{name}_256")]
        print(name, "pool max diff", float(np.abs(pooled - dst).max()))
        np.testing.assert_allclose(pooled, dst, rtol=1e-6, atol=1e-6, err_msg=name)


if __name__ == "__main__":
    main()
