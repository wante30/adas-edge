"""B-05L fair PyTorch timing: all models in one process, same tensors and clock state."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO


def parse_model(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    return label, Path(path)


def clocks() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu", "--format=csv,noheader"],
            text=True, timeout=20,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=parse_model, action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--batch-iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    single = torch.randn(1, 3, 192, 512, device=args.device)
    batch = torch.randn(args.batch, 3, 192, 512, device=args.device)
    result: dict[str, Any] = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "batch": args.batch,
                              "clocks_before": clocks(), "models": {}}
    for label, path in args.model:
        model = YOLO(str(path)).model.to(args.device).eval()
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(single)
            torch.cuda.synchronize()
            single_times = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                model(single)
                torch.cuda.synchronize()
                single_times.append((time.perf_counter() - started) * 1000.0)
            for _ in range(5):
                model(batch)
            torch.cuda.synchronize()
            batch_times = []
            for _ in range(args.batch_iterations):
                started = time.perf_counter()
                model(batch)
                torch.cuda.synchronize()
                batch_times.append((time.perf_counter() - started) * 1000.0)
        result["models"][label] = {
            "path": str(path.resolve()),
            "parameters": sum(value.numel() for value in model.state_dict().values()),
            "single_frame_ms_mean": float(np.mean(single_times)),
            "single_frame_ms_p50": float(np.percentile(single_times, 50)),
            "single_frame_ms_p95": float(np.percentile(single_times, 95)),
            "batch_ms_mean": float(np.mean(batch_times)),
            "batch_per_image_ms": float(np.mean(batch_times) / args.batch),
            "batch_images_per_second": float(args.batch * 1000.0 / np.mean(batch_times)),
        }
        del model
        torch.cuda.empty_cache()
    result["clocks_after"] = clocks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
