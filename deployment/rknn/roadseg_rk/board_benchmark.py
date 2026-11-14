"""Run one RKNN model on the RK3566 NPU and record real latency/resource data.

Run this file on the board, e.g.:
  python3 board_benchmark.py --model candidate_plus_fp16.rknn \
      --input-npy sample.npy --warmup 10 --runs 100 --json result.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import time
from pathlib import Path

import numpy as np


def read_text(path: str):
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def rss_bytes():
    try:
        with open("/proc/self/status", "r") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def cpu_times():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime, usage.ru_stime


def meminfo():
    result = {}
    try:
        with open("/proc/meminfo", "r") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                if key in ("MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached"):
                    result[key] = int(value.strip().split()[0]) * 1024
    except Exception:
        pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-npy", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--save-output", type=Path)
    args = parser.parse_args()

    if args.warmup < 0 or args.runs < 1:
        raise ValueError("warmup must be >= 0 and runs must be >= 1")

    image = np.load(str(args.input_npy))
    if image.ndim == 3:
        image = image[None, ...]
    if image.ndim != 4 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(
            "expected NHWC uint8 input, got shape=%r dtype=%s"
            % (image.shape, image.dtype)
        )

    from rknnlite.api import RKNNLite

    rknn = RKNNLite(verbose=False)
    started = time.perf_counter()
    if rknn.load_rknn(str(args.model)) != 0:
        raise RuntimeError("load_rknn failed")
    if rknn.init_runtime() != 0:
        raise RuntimeError("init_runtime failed")
    init_ms = (time.perf_counter() - started) * 1000.0

    freq_before = read_text("/sys/class/devfreq/fde40000.npu/cur_freq")
    load_before = read_text("/sys/class/devfreq/fde40000.npu/load")
    rss_after_init = rss_bytes()
    cpu_before = cpu_times()

    for _ in range(args.warmup):
        rknn.inference(inputs=[image], data_format="nhwc")

    timings = []
    outputs = None
    for _ in range(args.runs):
        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=[image], data_format="nhwc")
        timings.append((time.perf_counter() - t0) * 1000.0)

    cpu_after = cpu_times()
    wall = float(np.sum(timings)) / 1000.0
    cpu_seconds = (cpu_after[0] - cpu_before[0]) + (cpu_after[1] - cpu_before[1])
    cpu_percent = 100.0 * cpu_seconds / wall if wall > 0 else None

    freq_after = read_text("/sys/class/devfreq/fde40000.npu/cur_freq")
    load_after = read_text("/sys/class/devfreq/fde40000.npu/load")
    rss_after_runs = rss_bytes()

    output_info = []
    for output in outputs or []:
        arr = np.asarray(output)
        info = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "finite": bool(np.isfinite(arr).all()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }
        if arr.ndim == 4 and arr.shape[1] == 4:
            info["channel_mean"] = [float(arr[0, c].mean()) for c in range(4)]
            info["channel_min"] = [float(arr[0, c].min()) for c in range(4)]
            info["channel_max"] = [float(arr[0, c].max()) for c in range(4)]
        output_info.append(info)
    if args.save_output and outputs:
        np.save(str(args.save_output), np.asarray(outputs[0], dtype=np.float32))

    values = np.asarray(timings, dtype=np.float64)
    result = {
        "model": str(args.model),
        "model_size_bytes": args.model.stat().st_size,
        "input_shape": list(image.shape),
        "input_dtype": str(image.dtype),
        "warmup": args.warmup,
        "runs": args.runs,
        "init_ms": round(init_ms, 3),
        "latency_ms": {
            "mean": round(float(values.mean()), 3),
            "p50": round(float(np.percentile(values, 50)), 3),
            "p95": round(float(np.percentile(values, 95)), 3),
            "p99": round(float(np.percentile(values, 99)), 3),
            "min": round(float(values.min()), 3),
            "max": round(float(values.max()), 3),
            "stdev": round(float(values.std(ddof=1)), 3) if len(values) > 1 else 0.0,
        },
        "inference_fps": round(1000.0 / float(values.mean()), 3),
        "timings_ms": [round(float(v), 4) for v in timings],
        "cpu": {
            "user_seconds": round(cpu_after[0] - cpu_before[0], 4),
            "system_seconds": round(cpu_after[1] - cpu_before[1], 4),
            "percent_of_one_core": round(cpu_percent, 2) if cpu_percent else None,
        },
        "memory": {
            "rss_after_init": rss_after_init,
            "rss_after_runs": rss_after_runs,
            "system": meminfo(),
        },
        "npu": {
            "cur_freq_before": freq_before,
            "cur_freq_after": freq_after,
            "load_before": load_before,
            "load_after": load_after,
        },
        "outputs": output_info,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    rknn.release()


if __name__ == "__main__":
    main()
