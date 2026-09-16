"""Controlled latency benchmark for square vs rectangular YOLO input shapes.

``Results.speed`` is a per-batch profiler reading and comes out identical across
images, so it cannot answer "what does a 512x288 input actually cost on the
GPU". This tool measures wall-clock latency per image with an explicit CUDA
synchronisation around every call, on the same frozen holdout images, and records
the tensor shape the network actually received - which is the only way to tell a
real cost difference from a letterboxing artefact.

Reported per shape: mean/median/p95 wall ms per image, frames per second, the
observed model input shape, and the projected four-stream headroom.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def parse_shape(text: str) -> int | tuple[int, int]:
    """Return an Ultralytics ``imgsz``: an int, or ``(height, width)``."""
    clean = text.strip().lower().replace("×", "x").replace("*", "x")
    if "x" not in clean:
        return int(clean)
    width, height = (int(part) for part in clean.split("x", 1))
    return (height, width)


def describe_shape(shape: int | tuple[int, int]) -> str:
    return f"{shape}x{shape}" if isinstance(shape, int) else f"{shape[1]}x{shape[0]}"


def benchmark_shape(
    model,
    images: list[str],
    shape: int | tuple[int, int],
    repeats: int,
    warmup: int,
    device: str,
    torch,
) -> dict:
    observed: dict[str, tuple[int, ...]] = {}

    def capture(_module, args, _output=None):
        tensor = args[0] if args else None
        if hasattr(tensor, "shape"):
            observed[str(tuple(tensor.shape[1:]))] = tuple(tensor.shape[1:])

    handle = model.model.register_forward_pre_hook(capture)
    try:
        for index in range(warmup):
            model.predict(source=images[index % len(images)], imgsz=shape, conf=0.25, device=device, verbose=False)
        if device != "cpu":
            torch.cuda.synchronize()

        latencies: list[float] = []
        for index in range(repeats):
            source = images[index % len(images)]
            started = time.perf_counter()
            model.predict(source=source, imgsz=shape, conf=0.25, device=device, verbose=False)
            if device != "cpu":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000)
    finally:
        handle.remove()

    ordered = sorted(latencies)
    return {
        "requested": describe_shape(shape),
        "observed_input_shapes": sorted([list(key) for key in observed]),
        "frames": len(latencies),
        "mean_ms": statistics.fmean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": ordered[max(0, int(len(ordered) * 0.95) - 1)],
        "fps": 1000.0 / statistics.fmean(latencies),
        "four_stream_fps": 4 * 1000.0 / statistics.fmean(latencies),
        "four_stream_headroom_at_6fps": round(4 * 1000.0 / statistics.fmean(latencies) / 6.0, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--images", type=Path, required=True, help="directory of jpgs")
    parser.add_argument("--shapes", default="320,416,512x288", help="comma separated, e.g. 320,416,512x288")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    images = sorted(str(path) for path in args.images.glob("*.jpg"))
    if len(images) < args.warmup + 1:
        raise SystemExit(f"need at least {args.warmup + 1} images, found {len(images)}")

    model = YOLO(str(args.model))
    model.to(f"cuda:{args.device}" if args.device != "cpu" else "cpu")
    if args.device != "cpu":
        torch.backends.cudnn.benchmark = True

    results = []
    for text in [part.strip() for part in args.shapes.split(",") if part.strip()]:
        shape = parse_shape(text)
        entry = benchmark_shape(model, images, shape, args.repeats, args.warmup, args.device, torch)
        results.append(entry)
        print(
            f"{entry['requested']:>10}  mean {entry['mean_ms']:7.2f} ms  median {entry['median_ms']:7.2f}  "
            f"p95 {entry['p95_ms']:7.2f}  {entry['fps']:6.1f} FPS  input {entry['observed_input_shapes']}",
            flush=True,
        )

    report = {
        "model": str(args.model),
        "device": args.device,
        "repeats": args.repeats,
        "images_dir": str(args.images),
        "shapes": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}", file=sys.stdout)


if __name__ == "__main__":
    main()
