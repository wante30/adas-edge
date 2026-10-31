"""B-05L aggregate evaluation tables and consistency checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(label_path: str) -> tuple[str, dict[str, Any]]:
    label, path = label_path.split("=", 1)
    return label, json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", action="append", required=True, help="LABEL=eval.json")
    parser.add_argument("--meta", action="append", default=[], help="LABEL=run_meta.json")
    parser.add_argument("--baseline", default="yolo11n")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = dict(load(value) for value in args.eval)
    metas = dict(load(value) for value in args.meta)
    baseline = models[args.baseline]["metrics_roi"]
    rows = {}
    for label, data in models.items():
        metrics = data["metrics_roi"]
        if metrics["tp_sum_size_groups"] != metrics["tp"]:
            raise AssertionError(f"{label}: size-group TP {metrics['tp_sum_size_groups']} != {metrics['tp']}")
        if data["dataset"]["roi_gt"] != 150 or data["dataset"]["all_gt"] != 154:
            raise AssertionError(f"{label}: unexpected BDD50 GT counts")
        sizes = {name: {**value, "recall": value["tp"] / value["gt"] if value["gt"] else 0.0}
                 for name, value in metrics["size_groups"].items()}
        meta = metas.get(label)
        rows[label] = {
            "ap50": metrics["ap50"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
            "full_frame_recall": data["metrics_full_frame"]["recall"],
            "size_groups": sizes,
            "fp_budget": data["curves"]["fp_budget"],
            "mean_inference_ms": data["runtime"]["mean_inference_ms"],
            "p50_inference_ms": data["runtime"]["p50_inference_ms"],
            "p95_inference_ms": data["runtime"]["p95_inference_ms"],
            "parameters": data["model"]["parameters"],
            "size_bytes": data["model"]["size_bytes"],
            "color_accuracy_excluding_none": data["color"]["aggregate"]["accuracy_excluding_none"],
            "color_confusion": data["color"]["aggregate"]["confusion"],
            "backend": data["model"]["backend"],
            "device": data["model"]["device"],
            "onnx": (meta or {}).get("onnx"),
        }
        if label == args.baseline:
            rows[label]["delta_vs_baseline"] = None
        else:
            rows[label]["delta_vs_baseline"] = {
                "ap50": metrics["ap50"] - baseline["ap50"],
                "precision": metrics["precision"] - baseline["precision"],
                "recall": metrics["recall"] - baseline["recall"],
                "f1": metrics["f1"] - baseline["f1"],
                "tp": metrics["tp"] - baseline["tp"],
                "fp": metrics["fp"] - baseline["fp"],
            }
    payload = {"baseline": args.baseline, "models": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    order = [args.baseline] + [label for label in rows if label != args.baseline]
    print("| model | AP50 | P | R | F1 | TP/FP/FN | <8 | 8-16 | 16-32 | >32 | P50 ms | P95 ms | params |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label in order:
        row = rows[label]
        sizes = row["size_groups"]
        print(f"| {label} | {row['ap50']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
              f"{row['tp']}/{row['fp']}/{row['fn']} | {sizes['<8']['recall']:.4f} | {sizes['8-16']['recall']:.4f} | "
              f"{sizes['16-32']['recall']:.4f} | {sizes['>32']['recall']:.4f} | "
              f"{row['p50_inference_ms']:.2f} | {row['p95_inference_ms']:.2f} | {row['parameters']} |")
    print(json.dumps({label: {"delta_vs_baseline": rows[label]["delta_vs_baseline"]} for label in order if label != args.baseline}, indent=2))


if __name__ == "__main__":
    main()
