"""Probe RKNN conversion + simulator parity on one image."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ONNX = "/mnt/d/Datasets/adas_edge_training/b05n_final/b05n_p2_multiclass.onnx"
OUT = "/mnt/d/Datasets/adas_edge_training/rknn_b05o_test/b05n_p2_fp16.rknn"
SAMPLE = "/mnt/d/Datasets/adas_edge_training/b05n_bdd_multiclass/images/test/b1d22449-15fb948f.jpg"


def main() -> None:
    from rknn.api import RKNN
    import onnxruntime as ort

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(SAMPLE)
    if bgr is None:
        bgr = next(Path("/mnt/d/Datasets/adas_edge_training/b05n_bdd_multiclass/images/test").glob("*.jpg"))
        bgr = cv2.imread(str(bgr))
    rgb = cv2.cvtColor(cv2.resize(bgr, (512, 288), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3566")
    assert rknn.load_onnx(model=ONNX) == 0
    assert rknn.build(do_quantization=False) == 0
    assert rknn.export_rknn(OUT) == 0
    assert rknn.init_runtime() == 0
    outputs = rknn.inference(inputs=[rgb])
    print("rknn outputs", [tuple(o.shape) for o in outputs], flush=True)
    rknn.release()

    session = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    reference = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    print("onnx output", reference.shape)
    candidate = outputs[0]
    if candidate.shape != reference.shape:
        candidate = candidate.reshape(reference.shape)
    diff = np.abs(candidate.astype(np.float32) - reference)
    print("max abs diff", float(diff.max()), "mean", float(diff.mean()), flush=True)


if __name__ == "__main__":
    main()
