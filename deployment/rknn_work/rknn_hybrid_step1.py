"""Generate a hybrid-quantization config for a split YOLO RKNN model."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--proposal", action="store_true")
    ap.add_argument("--proposal-size", type=int, default=20)
    args = ap.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3566",
                quantized_method="channel")
    print("load", rknn.load_onnx(model=args.onnx), flush=True)
    print("step1", rknn.hybrid_quantization_step1(dataset=args.calib, proposal=args.proposal,
                                                  proposal_dataset_size=args.proposal_size), flush=True)
    rknn.release()
    print("files", sorted(p.name for p in workdir.iterdir()), flush=True)


if __name__ == "__main__":
    main()
