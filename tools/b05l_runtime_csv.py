"""Combine B-05L per-video runtime CSVs into logs/runtime.csv."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    summary = defaultdict(lambda: {"frames": 0, "inference_ms": []})
    for path in sorted((args.root / "logs" / "runtime").glob("*/*.csv")):
        model = path.parent.name
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["model"] = model
                rows.append(row)
                if row.get("inference_ms"):
                    summary[(model, row["video"])]["frames"] += 1
                    summary[(model, row["video"])]["inference_ms"].append(float(row["inference_ms"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "video", "frame", "video_time_ms", "backend", "schedule", "preprocess_ms",
              "inference_ms", "postprocess_ms", "offline_fps"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    result = {
        str(key): {
            "frames": value["frames"],
            "mean_inference_ms": float(np.mean(value["inference_ms"])) if value["inference_ms"] else None,
            "p50_inference_ms": float(np.percentile(value["inference_ms"], 50)) if value["inference_ms"] else None,
            "p95_inference_ms": float(np.percentile(value["inference_ms"], 95)) if value["inference_ms"] else None,
        }
        for key, value in sorted(summary.items())
    }
    (args.output.parent / "runtime_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.output.resolve()), "summary": result}, indent=2))


if __name__ == "__main__":
    main()
