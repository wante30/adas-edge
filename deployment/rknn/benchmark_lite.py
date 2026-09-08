"""Measure RKNN Lite raw inference with a preprocessed NHWC NumPy input."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-npy", type=Path, required=True, help="HWC or NHWC uint8 array")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("--warmup must be non-negative and --runs must be positive")

    image = np.load(args.input_npy)
    if image.ndim == 3:
        image = image[None, ...]
    if image.ndim != 4 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected NHWC uint8 input, got shape={image.shape}, dtype={image.dtype}")

    from rknnlite.api import RKNNLite
    try:
        import psutil
        process = psutil.Process(os.getpid())
    except ImportError:
        process = None

    rknn = RKNNLite(verbose=False)
    try:
        if rknn.load_rknn(str(args.model)) != 0:
            raise RuntimeError("load_rknn failed")
        if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
            raise RuntimeError("init_runtime failed")
        rss_after_init = process.memory_info().rss if process else None
        for _ in range(args.warmup):
            rknn.inference(inputs=[image], data_format="nhwc")
        timings: list[float] = []
        for _ in range(args.runs):
            started = time.perf_counter()
            outputs = rknn.inference(inputs=[image], data_format="nhwc")
            timings.append((time.perf_counter() - started) * 1000)
        rss_after_runs = process.memory_info().rss if process else None
    finally:
        rknn.release()

    values = np.asarray(timings)
    result = {
        "model": str(args.model),
        "input_shape": list(image.shape),
        "input_dtype": str(image.dtype),
        "runs": args.runs,
        "latency_ms": {
            "mean": round(float(values.mean()), 3),
            "p50": round(float(np.percentile(values, 50)), 3),
            "p95": round(float(np.percentile(values, 95)), 3),
            "min": round(float(values.min()), 3),
            "max": round(float(values.max()), 3),
        },
        "inference_fps": round(1000.0 / float(values.mean()), 3),
        "process_rss_bytes": {"after_init": rss_after_init, "after_runs": rss_after_runs},
        "outputs": [
            {"shape": list(output.shape), "dtype": str(output.dtype), "finite": bool(np.isfinite(output).all())}
            for output in outputs
        ],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json_output:
        args.json_output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
