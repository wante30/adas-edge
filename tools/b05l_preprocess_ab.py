"""B-05L preprocessing A/B summary (letterbox vs direct resize)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(value: str) -> tuple[str, dict]:
    label, path = value.split("=", 1)
    return label, json.loads(Path(path).read_text(encoding="utf-8"))


def row(label: str, preprocess: str, data: dict) -> dict:
    metrics = data["metrics_roi"]
    return {
        "model": label,
        "preprocess": preprocess,
        "ap50": metrics["ap50"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
        "size_groups": metrics["size_groups"],
        "mean_inference_ms": data["runtime"]["mean_inference_ms"],
        "p95_inference_ms": data["runtime"]["p95_inference_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", action="append", required=True,
                        help="LABEL:letterbox=path or LABEL:resize=path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for value in args.eval:
        prefix, path = value.split("=", 1)
        label, preprocess = prefix.split(":", 1)
        _, data = load(f"{label}={path}")
        rows.append(row(label, preprocess, data))
    payload = {"rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("| model | preprocess | AP50 | P | R | TP/FP/FN | 8-16 recall |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for item in rows:
        print(f"| {item['model']} | {item['preprocess']} | {item['ap50']:.4f} | {item['precision']:.4f} | "
              f"{item['recall']:.4f} | {item['tp']}/{item['fp']}/{item['fn']} | "
              f"{item['size_groups']['8-16']['tp']}/{item['size_groups']['8-16']['gt']} |")


if __name__ == "__main__":
    main()
