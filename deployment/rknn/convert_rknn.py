"""Convert a static ONNX model into an RK3566 RKNN model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def check(status: int, step: str) -> None:
    if status != 0:
        raise RuntimeError(f"{step} failed with RKNN status {status}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--precision",
        choices=("fp16", "int8"),
        default="fp16",
        help="FP16 is the verified default; INT8 requires semantic-output validation.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Calibration list required only for --precision int8.",
    )
    args = parser.parse_args(argv)
    if args.precision == "int8" and args.dataset is None:
        parser.error("--dataset is required when --precision int8")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths_to_check = [args.onnx]
    if args.dataset is not None:
        paths_to_check.append(args.dataset)
    for path in paths_to_check:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    try:
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3566")
        check(rknn.load_onnx(model=str(args.onnx)), "load_onnx")
        if args.precision == "int8":
            check(rknn.build(do_quantization=True, dataset=str(args.dataset)), "build")
        else:
            check(rknn.build(do_quantization=False), "build")
        check(rknn.export_rknn(str(args.output)), "export_rknn")
    finally:
        rknn.release()
    print(f"RKNN exported ({args.precision}): {args.output}")


if __name__ == "__main__":
    main()
