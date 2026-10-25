#!/usr/bin/env bash
source /home/jingyuwang/.venvs/rknn232/bin/activate
python - <<'PY'
import onnx
paths = [
 '/mnt/d/Datasets/adas_edge_training/b03c_rknn_validation_20261021/bdd_yolov8n_512x288.onnx',
 '/mnt/d/Datasets/adas_edge_training/b05l_runs/v8n_p2/weights/best.onnx',
 '/mnt/d/Datasets/adas_edge_training/b05n_final/b05n_p2_multiclass.onnx',
]
for p in paths:
    m = onnx.load(p)
    ins = [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    outs = [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
    print(p.split('/')[-1], 'IN', ins, 'OUT', outs, 'nodes', len(m.graph.node))
PY
