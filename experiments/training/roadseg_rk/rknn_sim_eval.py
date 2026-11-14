"""RKNN Toolkit x86 simulator PTQ comparison against ONNX FP32.

This is a software/simulator check, not RK3566 board latency or board accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rknn.api import RKNN


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--eval-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--algorithm", default="normal", choices=("normal", "mmse", "kl_divergence"))
    p.add_argument("--method", default="channel", choices=("layer", "channel"))
    p.add_argument("--optimization-level", type=int, default=2)
    args = p.parse_args()

    rows = json.loads(Path(args.eval_manifest).read_text(encoding="utf-8"))
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3566",
                quantized_algorithm=args.algorithm, quantized_method=args.method,
                optimization_level=args.optimization_level)
    assert rknn.load_onnx(model=args.onnx) == 0
    assert rknn.build(do_quantization=True, dataset=args.dataset) == 0
    assert rknn.init_runtime() == 0  # simulator

    maes, maxes = [], []
    tp = fp = fn = 0
    tp_m = fp_m = fn_m = 0
    for row in rows:
        path = row["input"]
        if len(path) > 1 and path[1] == ":":
            path = "/mnt/" + path[0].lower() + path[2:].replace("\\", "/")
        x = np.load(path)  # NHWC uint8
        ref = sess.run(None, {"rgb": x.transpose(0, 3, 1, 2).astype(np.float32) / 255.0})[0]
        got = np.asarray(rknn.inference(inputs=[x], data_format="nhwc")[0], np.float32)
        mae = float(np.mean(np.abs(ref - got)))
        mx = float(np.max(np.abs(ref - got)))
        maes.append(mae); maxes.append(mx)
        pr = 1 / (1 + np.exp(-ref[0]))
        pg = 1 / (1 + np.exp(-got[0]))
        for c in range(pr.shape[0]):
            a = (pr[c] >= 0.5); b = (pg[c] >= 0.5)
            tp += int(np.sum(a & b)); fp += int(np.sum(b & ~a)); fn += int(np.sum(a & ~b))
        if pr.shape[0] >= 2:
            a = (pr[0] >= 0.5) | (pr[1] >= 0.5)
            b = (pg[0] >= 0.5) | (pg[1] >= 0.5)
            tp_m += int(np.sum(a & b)); fp_m += int(np.sum(b & ~a)); fn_m += int(np.sum(a & ~b))
    rknn.release()
    report = dict(onnx=str(Path(args.onnx).resolve()), algorithm=args.algorithm, method=args.method,
                  optimization_level=args.optimization_level, samples=len(rows),
                  logits_mae_mean=float(np.mean(maes)), logits_mae_max=float(np.max(maes)),
                  logits_abs_max=float(np.max(maxes)),
                  mask_iou_all_classes=float(tp / max(1, tp + fp + fn)),
                  mask_iou_merged_drivable=float(tp_m / max(1, tp_m + fp_m + fn_m)),
                  note="RKNN x86 simulator; not board accuracy and not board latency")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
