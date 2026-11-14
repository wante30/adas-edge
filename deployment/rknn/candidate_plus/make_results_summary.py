"""Collect the small, citable result JSON for the RK3566 Candidate-plus run."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    benchmarks = {}
    for key, name in (
        ("fp16", "fp16_bench.json"),
        ("int8_ptq", "int8_bench.json"),
        ("int8_qat", "qat_opt2_bench.json"),
    ):
        path = root / "logs" / name
        if path.exists():
            item = read(path)
            benchmarks[key] = {
                "model": item["model"],
                "latency_ms": item["latency_ms"],
                "inference_fps": item["inference_fps"],
                "npu": item["npu"],
                "runs": item["runs"],
                "warmup": item["warmup"],
            }

    video = None
    for name in ("video_full2.json", "video_full.json"):
        path = root / "logs" / name
        if path.exists():
            item = read(path)
            video = {
                "file": name,
                "source": item["source"],
                "processed_frames": item["processed_frames"],
                "resolution": item["resolution"],
                "fps": item["fps"],
                "wall_seconds": item["wall_seconds"],
                "pipeline_fps_wall": item["pipeline_fps_wall"],
                "pipeline_fps_sum_of_mean_stages": item["pipeline_fps_sum_of_mean_stages"],
                "stages_ms": {k: v["mean"] for k, v in item["stages_ms"].items()},
                "encoder": item["encoder"],
            }
            break

    accuracy = None
    path = root / "eval" / "accuracy" / "accuracy.json"
    if path.exists():
        accuracy = read(path)["summary"]

    concurrent = None
    path = root / "logs" / "concurrent_p2.json"
    if path.exists():
        item = read(path)
        concurrent = {
            "p2_only": item["p2_only"],
            "candidate_only": item["candidate_only"],
            "pair": item["pair"],
            "cpu_percent_of_one_core": item["cpu_percent_of_one_core"],
            "rss": item["rss"],
        }

    video_path = root / "candidate_plus_rk3566_overlay.mp4"
    if video is not None and video_path.exists():
        video["output_bytes"] = video_path.stat().st_size
        video["output_sha256"] = sha256(video_path)

    models = {}
    for name in (
        "candidate_plus_fp16.rknn",
        "candidate_plus_int8.rknn",
        "candidate_plus_int8_qat.rknn",
    ):
        path = root / "models" / name
        if path.exists():
            models[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    payload = {
        "date": "2026-11-14",
        "board": {
            "name": "EmbedFire LubanCat-1 RK3566",
            "npu_driver": "0.8.2",
            "librknnrt": "2.3.2",
            "npu_freq_hz": 600000000,
            "toolkit": "rknn-toolkit2 2.3.2",
        },
        "models": models,
        "benchmarks": benchmarks,
        "video": video,
        "accuracy_summary": accuracy,
        "concurrent_yolo_p2": concurrent,
        "artifacts": {
            "final_model": "models/candidate_plus_int8_qat.rknn",
            "final_video": "candidate_plus_rk3566_overlay.mp4",
            "report": "REPORT_CN.md",
        },
    }
    out = root / "results_summary.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
