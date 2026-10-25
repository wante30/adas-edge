"""B-05N latency: BDD main, full-frame P2, B05L specialist, dual total."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def clocks():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,clocks.sm,power.draw", "--format=csv,noheader"], text=True, timeout=20).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def parse_model(value: str) -> tuple[str, Path, int, int]:
    # format: label=path,height,width
    label, rest = value.split("=", 1)
    path, h, w = rest.rsplit(",", 2)
    return label, Path(path), int(h), int(w)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=parse_model, action="append", required=True,
                        help="label=path,height,width")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from ultralytics import YOLO

    result = {"gpu": clocks(), "models": {}}
    for label, path, height, width in args.model:
        model = YOLO(str(path)).model.cuda().eval()
        x = torch.randn(1, 3, height, width, device="cuda")
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(x)
            torch.cuda.synchronize()
            times = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                model(x)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - started) * 1000.0)
        result["models"][label] = {
            "path": str(path), "input": [1, 3, height, width],
            "mean_ms": float(np.mean(times)), "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        del model
        torch.cuda.empty_cache()
    if "bdd_main" in result["models"] and "specialist" in result["models"]:
        result["dual_sequential_p50_ms"] = result["models"]["bdd_main"]["p50_ms"] + result["models"]["specialist"]["p50_ms"]
        result["dual_sequential_p95_ms"] = result["models"]["bdd_main"]["p95_ms"] + result["models"]["specialist"]["p95_ms"]
    result["gpu_after"] = clocks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
