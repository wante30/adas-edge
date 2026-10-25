#!/usr/bin/env bash
set -e
source /home/jingyuwang/.venvs/rknn232/bin/activate
python - <<'PY'
import onnx, numpy
print('numpy', numpy.__version__)
m = onnx.load('/mnt/d/Datasets/adas_edge_training/b05n_final/b05n_p2_multiclass.onnx')
print('loaded', m.ir_version, len(m.graph.node))
print('input', m.graph.input[0].name, [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim])
PY
