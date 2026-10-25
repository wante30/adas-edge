"""Inspect one RKNN build (fp16/int8) vs ONNX on a sample image."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--precision", choices=("fp16", "int8"), default="fp16")
    ap.add_argument("--calib", default="/mnt/d/Datasets/adas_edge_training/rknn_b05o/calib_full")
    ap.add_argument("--image-index", type=int, default=200)
    ap.add_argument("--method", choices=("channel", "layer"), default="channel")
    ap.add_argument("--algorithm", default="normal")
    args = ap.parse_args()

    from rknn.api import RKNN
    import onnxruntime as ort

    listing = None
    if args.precision == "int8":
        calib = sorted(Path(args.calib).glob("*.jpg"))
        path = Path("/mnt/d/Datasets/adas_edge_training/rknn_b05o/debug_calib.txt")
        path.write_text("\n".join(str(p) for p in calib) + "\n", encoding="utf-8")
        listing = str(path)

    image = sorted(Path("/mnt/d/Datasets/adas_edge_training/b05n_bdd_multiclass/images/test").glob("*.jpg"))[args.image_index]
    bgr = cv2.imread(str(image))
    rgb = cv2.cvtColor(cv2.resize(bgr, (512, 288), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3566",
                quantized_method=args.method, quantized_algorithm=args.algorithm)
    print("load", rknn.load_onnx(model=args.onnx))
    print("build", rknn.build(do_quantization=args.precision == "int8", dataset=listing))
    print("init", rknn.init_runtime())
    outputs = rknn.inference(inputs=[np.ascontiguousarray(rgb[None])])
    rknn.release()
    rk = outputs[0]

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    on = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    print("shapes", rk.shape, on.shape)
    rk2 = rk.reshape(on.shape)
    for name, arr in (("rknn", rk2), ("onnx", on)):
        print(name, "cls min/max", float(arr[0, 4:].min()), float(arr[0, 4:].max()),
              "nonzero", int((arr[0, 4:] > 0.01).sum()),
              "box min/max", float(arr[0, :4].min()), float(arr[0, :4].max()))
    print("cls corr", float(np.corrcoef(rk2[0, 4:].ravel(), on[0, 4:].ravel())[0, 1]))
    print("box corr", float(np.corrcoef(rk2[0, :4].ravel(), on[0, :4].ravel())[0, 1]))


if __name__ == "__main__":
    main()
