"""Measure YOLO-P2 and Candidate-plus NPU inference alone and interleaved.

Both RKNNLite runtimes stay loaded in one process, so this tests the real
single-NPU sharing behavior rather than adding theoretical latencies.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np


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


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
        "fps_from_mean": round(1000.0 / float(arr.mean()), 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-model", type=Path, required=True)
    parser.add_argument("--p2-height", type=int, default=288)
    parser.add_argument("--p2-width", type=int, default=512)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    from rknnlite.api import RKNNLite

    candidate_input = np.load(str(args.candidate_input))
    if candidate_input.ndim == 3:
        candidate_input = candidate_input[None, ...]
    p2_input = np.zeros((1, args.p2_height, args.p2_width, 3), np.uint8)

    p2 = RKNNLite(verbose=False)
    cand = RKNNLite(verbose=False)
    if p2.load_rknn(str(args.p2_model)) != 0:
        raise RuntimeError("load P2 failed")
    if p2.init_runtime() != 0:
        raise RuntimeError("init P2 failed")
    if cand.load_rknn(str(args.candidate_model)) != 0:
        raise RuntimeError("load Candidate-plus failed")
    if cand.init_runtime() != 0:
        raise RuntimeError("init Candidate-plus failed")

    for _ in range(args.warmup):
        p2.inference(inputs=[p2_input], data_format="nhwc")
        cand.inference(inputs=[candidate_input], data_format="nhwc")

    cpu0 = cpu_times()
    rss0 = rss_bytes()

    p2_only = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        p2.inference(inputs=[p2_input], data_format="nhwc")
        p2_only.append((time.perf_counter() - t0) * 1000.0)

    cand_only = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        cand.inference(inputs=[candidate_input], data_format="nhwc")
        cand_only.append((time.perf_counter() - t0) * 1000.0)

    pair_total = []
    p2_in_pair = []
    cand_in_pair = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        a0 = time.perf_counter()
        p2.inference(inputs=[p2_input], data_format="nhwc")
        a1 = time.perf_counter()
        cand.inference(inputs=[candidate_input], data_format="nhwc")
        a2 = time.perf_counter()
        p2_in_pair.append((a1 - a0) * 1000.0)
        cand_in_pair.append((a2 - a1) * 1000.0)
        pair_total.append((a2 - t0) * 1000.0)

    cpu1 = cpu_times()
    rss1 = rss_bytes()
    wall = float(np.sum(pair_total)) / 1000.0
    cpu_seconds = (cpu1[0] - cpu0[0]) + (cpu1[1] - cpu0[1])
    result = {
        "p2_model": str(args.p2_model),
        "candidate_model": str(args.candidate_model),
        "p2_input_shape": list(p2_input.shape),
        "candidate_input_shape": list(candidate_input.shape),
        "warmup": args.warmup,
        "runs": args.runs,
        "p2_only": summarize(p2_only),
        "candidate_only": summarize(cand_only),
        "pair": {
            "total": summarize(pair_total),
            "p2": summarize(p2_in_pair),
            "candidate": summarize(cand_in_pair),
            "effective_total_fps": round(2000.0 / float(np.mean(pair_total)), 3),
        },
        "cpu_percent_of_one_core": round(100.0 * cpu_seconds / wall, 2) if wall else None,
        "rss": {"before": rss0, "after": rss1},
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    p2.release()
    cand.release()


if __name__ == "__main__":
    main()
