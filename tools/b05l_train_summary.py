"""Summarize a completed B-05L training run from results.csv and run metadata."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.run / "results.csv").open(encoding="utf-8")))
    meta = json.loads((args.run / "b05l_run_meta.json").read_text(encoding="utf-8"))
    numeric = [{key.strip(): float(value) for key, value in row.items()} for row in rows]
    fitness_key = "fitness" if "fitness" in numeric[0] else "metrics/mAP50-95(B)"
    best = max(numeric, key=lambda row: row[fitness_key])
    result = {
        "run": str(args.run.resolve()),
        "arch": meta["arch"],
        "epochs_requested": meta["epochs"],
        "epochs_completed": len(numeric),
        "batch": meta["batch"],
        "imgsz": meta["imgsz"],
        "seed": meta["seed"],
        "parameters": meta["parameters"],
        "transfer": meta["transfer"],
        "best_epoch": int(best["epoch"]),
        "best_fitness": best[fitness_key],
        "best_metrics": {key: value for key, value in best.items() if key.startswith("metrics/")},
        "final_metrics": {key: numeric[-1][key] for key in numeric[-1] if key.startswith("metrics/")},
        "train_seconds": numeric[-1]["time"],
        "best": meta["best"],
        "best_sha256": meta["best_sha256"],
        "onnx": meta.get("onnx"),
        "gpu": meta["gpu"],
        "torch": meta["torch"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
