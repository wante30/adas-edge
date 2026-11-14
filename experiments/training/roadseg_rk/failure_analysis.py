"""Rank real failure cases from per-image evaluation JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args()
    ev = Path(args.eval_dir)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    result = {}

    bdd = load(ev / "per_image_bdd_official_val.jsonl")
    # Drivable over-segmentation: many false positives relative to GT.
    def drivable_fp(r):
        c = r["classes"].get("drivable_area") or {}
        denom = (c.get("tp", 0) + c.get("fp", 0) + c.get("fn", 0))
        return c.get("fp", 0) / denom if denom else -1
    def drivable_fn(r):
        c = r["classes"].get("drivable_area") or {}
        denom = (c.get("tp", 0) + c.get("fp", 0) + c.get("fn", 0))
        return c.get("fn", 0) / denom if denom else -1
    def lane_fp(name):
        def score(r):
            c = r["classes"].get(name) or {}
            return c.get("fp", 0) - 0.25 * c.get("tp", 0)
        return score
    def lane_fn(name):
        def score(r):
            c = r["classes"].get(name) or {}
            return c.get("fn", 0) - 0.25 * c.get("tp", 0)
        return score

    categories = {
        "bdd_drivable_fp": sorted(bdd, key=drivable_fp, reverse=True)[:args.top],
        "bdd_drivable_fn": sorted(bdd, key=drivable_fn, reverse=True)[:args.top],
        "bdd_solid_fp": sorted(bdd, key=lane_fp("solid_lane"), reverse=True)[:args.top],
        "bdd_dashed_fp": sorted(bdd, key=lane_fp("dashed_lane"), reverse=True)[:args.top],
        "bdd_solid_fn": sorted(bdd, key=lane_fn("solid_lane"), reverse=True)[:args.top],
        "bdd_dashed_fn": sorted(bdd, key=lane_fn("dashed_lane"), reverse=True)[:args.top],
    }
    rlmd = load(ev / "per_image_rlmd_val.jsonl")
    cross = []
    for r in rlmd:
        c = r["classes"].get("crosswalk") or {}
        cross.append((c.get("fp", 0), c.get("fn", 0), r))
    categories["rlmd_crosswalk_fp"] = [r for fp, fn, r in sorted(cross, key=lambda x: x[0], reverse=True) if fp > 0][:args.top]
    categories["rlmd_crosswalk_fn"] = [r for fp, fn, r in sorted(cross, key=lambda x: x[1], reverse=True) if fn > 0][:args.top]

    for name, rows in categories.items():
        result[name] = [
            dict(id=r["id"], source=r["source"],
                 drivable_iou=(r.get("drivable_merged") or {}).get("iou"),
                 classes={k: {m: v.get(m) for m in ("iou", "f1", "tp", "fp", "fn")} for k, v in r["classes"].items()})
            for r in rows
        ]
        (out / f"{name}_ids.txt").write_text("\n".join(r["id"] for r in rows) + "\n", encoding="utf-8")
    (out / "failures.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, rows in result.items():
        print(name, len(rows), [r["id"] for r in rows[:3]])


if __name__ == "__main__":
    main()
