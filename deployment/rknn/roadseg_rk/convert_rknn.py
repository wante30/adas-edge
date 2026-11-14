"""Convert a static Candidate-plus ONNX model to RK3566 RKNN.

Tested with rknn-toolkit2 2.3.2 / Python 3.10 on x86_64.
FP16 is the graph/operator compatibility check.  INT8 uses a dataset.txt of
letterboxed RGB uint8 .npy tensors (shape [1,192,320,3]).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def check(status: int, step: str) -> None:
    if status != 0:
        raise RuntimeError("%s failed with RKNN status %s" % (step, status))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--precision",
        choices=("fp16", "int8"),
        default="fp16",
        help="fp16 is unquantized; int8 requires --dataset",
    )
    parser.add_argument("--dataset", type=Path, help="dataset.txt for INT8 calibration")
    parser.add_argument(
        "--quantized-algorithm",
        default="normal",
        choices=("normal", "mmse", "kl_divergence"),
    )
    parser.add_argument(
        "--quantized-method", default="channel", choices=("layer", "channel")
    )
    parser.add_argument("--target-platform", default="rk3566")
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=3,
        choices=(0, 1, 2, 3),
        help="use 2 for QAT QDQ models to avoid accuracy-affecting graph passes",
    )
    parser.add_argument(
        "--rgb2bgr",
        action="store_true",
        help="only relevant for image datasets; .npy ignores this flag",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.precision == "int8" and args.dataset is None:
        parser.error("--dataset is required when --precision int8")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    if args.dataset is not None and not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from rknn.api import RKNN

    rknn = RKNN(verbose=args.verbose)
    try:
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=args.target_platform,
            quantized_algorithm=args.quantized_algorithm,
            quantized_method=args.quantized_method,
            quant_img_RGB2BGR=args.rgb2bgr,
            float_dtype="float16",
            optimization_level=args.optimization_level,
        )
        check(rknn.load_onnx(model=str(args.onnx)), "load_onnx")
        if args.precision == "int8":
            check(
                rknn.build(do_quantization=True, dataset=str(args.dataset)),
                "build(int8)",
            )
        else:
            check(rknn.build(do_quantization=False), "build(fp16)")
        check(rknn.export_rknn(str(args.output)), "export_rknn")
    finally:
        rknn.release()
    print("RKNN exported (%s): %s" % (args.precision, args.output))


if __name__ == "__main__":
    main()
