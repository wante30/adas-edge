"""Validate split-output RKNN (boxes/logits) against the unsplit ONNX."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-onnx", required=True)
    ap.add_argument("--reference-onnx", required=True)
    ap.add_argument("--precision", choices=("fp16", "int8"), default="int8")
    ap.add_argument("--calib", default="/mnt/d/Datasets/adas_edge_training/rknn_b05o/calib_full")
    ap.add_argument("--image-index", type=int, default=200)
    args = ap.parse_args()

    from rknn.api import RKNN
    import onnxruntime as ort

    listing = None
    if args.precision == "int8":
        calib = sorted(Path(args.calib).glob("*.jpg"))
        path = Path("/mnt/d/Datasets/adas_edge_training/rknn_b05o/debug_split_calib.txt")
        path.write_text("\n".join(str(p) for p in calib) + "\n", encoding="utf-8")
        listing = str(path)

    image = sorted(Path("/mnt/d/Datasets/adas_edge_training/b05n_bdd_multiclass/images/test").glob("*.jpg"))[args.image_index]
    bgr = cv2.imread(str(image))
    rgb = cv2.cvtColor(cv2.resize(bgr, (512, 288), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3566",
                quantized_method="channel")
    print("load", rknn.load_onnx(model=args.split_onnx))
    print("build", rknn.build(do_quantization=args.precision == "int8", dataset=listing))
    print("init", rknn.init_runtime())
    outs = rknn.inference(inputs=[np.ascontiguousarray(rgb[None])])
    rknn.release()
    print("out shapes", [o.shape for o in outs])
    boxes, scores = outs[0], outs[1]
    if boxes.shape[0] == 1:
        boxes = boxes[0]
    if scores.shape[0] == 1:
        scores = scores[0]
    probs = 1.0 / (1.0 + np.exp(-scores.astype(np.float64)))
    sealed = np.concatenate([boxes.astype(np.float32), probs.astype(np.float32)], axis=0)[None]

    session = ort.InferenceSession(args.reference_onnx, providers=["CPUExecutionProvider"])
    ref = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    print("cls min/max", float(sealed[0, 4:].min()), float(sealed[0, 4:].max()),
          "nonzero", int((sealed[0, 4:] > 0.01).sum()))
    print("ref cls min/max", float(ref[0, 4:].min()), float(ref[0, 4:].max()),
          "nonzero", int((ref[0, 4:] > 0.01).sum()))
    print("cls corr", float(np.corrcoef(sealed[0, 4:].ravel(), ref[0, 4:].ravel())[0, 1]))
    print("box corr", float(np.corrcoef(sealed[0, :4].ravel(), ref[0, :4].ravel())[0, 1]))
    print("box max diff", float(np.abs(sealed[0, :4] - ref[0, :4]).max()))
    print("cls max diff", float(np.abs(sealed[0, 4:] - ref[0, 4:]).max()))


if __name__ == "__main__":
    main()
