"""Build a hybrid-quantized RKNN model and evaluate mAP50 on the BDD holdout."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rknn_eval import CLASSES, decode_multiclass, load_gt, prepare_full  # noqa: E402
from tools.b05k_common import evaluate_predictions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="step1 .model file")
    ap.add_argument("--data", required=True, help="step1 .data file")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--export")
    ap.add_argument("--reference-onnx", required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--offset", type=int, default=200)
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from rknn.api import RKNN
    import onnxruntime as ort

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3566",
                quantized_method="channel")
    print("step2", rknn.hybrid_quantization_step2(model_input=args.model, data_input=args.data,
                                                  model_quantization_cfg=args.cfg), flush=True)
    if args.export:
        print("export", rknn.export_rknn(args.export), flush=True)
    print("init", rknn.init_runtime(), flush=True)

    session = ort.InferenceSession(args.reference_onnx, providers=["CPUExecutionProvider"])
    names = sorted(args.images.glob("*.jpg"))[args.offset: args.offset + args.limit]
    gt = {name: {} for name in CLASSES}
    preds_rknn = {name: {} for name in CLASSES}
    preds_onnx = {name: {} for name in CLASSES}
    for source in names:
        bgr = cv2.imread(str(source))
        rgb, tensor = prepare_full(bgr)
        outs = rknn.inference(inputs=[rgb])
        boxes, scores = outs[0], outs[1]
        if boxes.ndim == 3:
            boxes = boxes[0]
        if scores.ndim == 3:
            scores = scores[0]
        probs = 1.0 / (1.0 + np.exp(-scores.astype(np.float64)))
        sealed = np.concatenate([boxes.astype(np.float32), probs.astype(np.float32)], axis=0)[None]
        on = session.run(None, {session.get_inputs()[0].name: tensor})[0]

        rk_boxes = decode_multiclass(sealed, args.conf, args.iou)
        on_boxes = decode_multiclass(on, args.conf, args.iou)
        for name in CLASSES:
            preds_rknn[name][source.name] = rk_boxes[name]
            preds_onnx[name][source.name] = on_boxes[name]
            gt[name][source.name] = load_gt(args.labels, source.name, 1280, 720, CLASSES.index(name))
    rknn.release()

    aps = {"rknn": {}, "onnx": {}}
    for name in CLASSES:
        if sum(len(v) for v in gt[name].values()) == 0:
            continue
        for tag, table in (("rknn", preds_rknn), ("onnx", preds_onnx)):
            aps[tag][name] = evaluate_predictions(gt[name], table[name], conf=0.25, iou=0.5, ap_floor=0.001)["ap50"]
    result = {"images": len(names), "cfg": args.cfg, "map50": aps,
              "map50_mean": {tag: float(np.mean(list(v.values()))) for tag, v in aps.items()}}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
