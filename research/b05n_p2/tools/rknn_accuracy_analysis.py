"""Run RKNN accuracy_analysis on the INT8 P2 model and save per-layer errors."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--images", required=True, nargs="+")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3566",
                quantized_method="channel")
    print("load", rknn.load_onnx(model=args.onnx))
    print("build", rknn.build(do_quantization=True, dataset=args.calib))
    print("init", rknn.init_runtime())
    for index, path in enumerate(args.images):
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(cv2.resize(bgr, (512, 288), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        out = Path(args.output) / f"image_{index}"
        out.mkdir(parents=True, exist_ok=True)
        print("analysis", index, path, flush=True)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        status = rknn.accuracy_analysis(inputs=[tensor], output_dir=str(out))
        print("status", status, flush=True)
    rknn.release()


if __name__ == "__main__":
    main()
