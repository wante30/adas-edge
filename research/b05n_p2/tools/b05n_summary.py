"""B-05N comparison tables from Ultralytics holdout JSON and BDD50 TL JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(value: str):
    label, path = value.split("=", 1)
    return label, json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", action="append", required=True, help="LABEL=1000-image eval JSON")
    ap.add_argument("--tl", action="append", default=[], help="LABEL=BDD50 TL eval JSON")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    evals = dict(load(v) for v in args.eval)
    tls = dict(load(v) for v in args.tl)
    rows = {}
    for label, data in evals.items():
        overall = data["models"][label]["overall"]
        per = data["models"][label]["per_class"]
        row = {"overall": overall, "per_class": per}
        if label in tls:
            row["tl"] = {"roi": tls[label]["metrics_roi"], "full": tls[label]["metrics_full"]}
        rows[label] = row
    payload = {"models": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("| model | mAP50 | mAP50-95 | car AP50 | person AP50 | rider AP50 | TL AP50 | TL Recall | 8-16 TL Recall |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, row in rows.items():
        per = row["per_class"]
        tl = row.get("tl", {}).get("roi", {})
        size = tl.get("size_groups", {}).get("8-16", {})
        tl_recall = size.get("tp", 0) / size.get("gt", 1) if size else 0.0
        print(f"| {label} | {row['overall']['map50']:.4f} | {row['overall']['map50_95']:.4f} | "
              f"{per['car']['ap50']:.4f} | {per['person']['ap50']:.4f} | {per['rider']['ap50']:.4f} | "
              f"{tl.get('ap50', 0):.4f} | {tl.get('recall', 0):.4f} | {tl_recall:.4f} |")


if __name__ == "__main__":
    main()
