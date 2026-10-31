"""Print the tail of the YOLO ONNX graph to understand the output concat."""
from __future__ import annotations

import onnx
from onnx import shape_inference

PATH = "/mnt/d/Datasets/adas_edge_training/b05n_final/b05n_p2_multiclass.onnx"

model = shape_inference.infer_shapes(onnx.load(PATH))
graph = model.graph
producers = {out: node for node in graph.node for out in node.output}
shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
          for v in (*graph.input, *graph.value_info, *graph.output)}
out = graph.output[0].name
node = producers[out]
print("output", out, shapes[out], "producer", node.op_type, "inputs", len(node.input))
for name in node.input:
    inner = producers.get(name)
    print("  input", name, shapes.get(name), "producer", inner.op_type if inner else None,
          "n_in", len(inner.input) if inner else None)
    if inner is not None and inner.op_type == "Concat":
        for sub in inner.input:
            subnode = producers.get(sub)
            print("    sub", sub, shapes.get(sub), "producer", subnode.op_type if subnode else None,
                  "n_in", len(subnode.input) if subnode else None)
