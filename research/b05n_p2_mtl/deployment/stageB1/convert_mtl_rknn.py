"""Convert the B-1 multi-task ONNX to RK3566 FP16 or INT8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--precision", choices=("fp16", "int8"), required=True)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--details", default="")
    ap.add_argument("--algorithm", default="normal")
    ap.add_argument("--method", default="channel")
    args = ap.parse_args()

    if args.precision == "int8" and not args.dataset:
        raise SystemExit("--dataset required for int8")

    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    try:
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform="rk3566",
            quantized_algorithm=args.algorithm,
            quantized_method=args.method,
        )
        ret = rknn.load_onnx(model=args.onnx)
        if ret != 0:
            raise RuntimeError(f"load_onnx failed: {ret}")
        if args.precision == "int8":
            ret = rknn.build(do_quantization=True, dataset=args.dataset)
        else:
            ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"build failed: {ret}")
        ret = rknn.export_rknn(args.output)
        if ret != 0:
            raise RuntimeError(f"export_rknn failed: {ret}")
        details = rknn.get_output_details()
        if args.details:
            Path(args.details).write_text(json.dumps(details, indent=2))
        print(json.dumps({"output": args.output, "precision": args.precision,
                          "outputs": details}, indent=2))
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
