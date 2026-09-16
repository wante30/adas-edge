"""Build a balanced, reproducible YOLO subset from licensed BDD100K train data.

The script only reads ``images/100k/train`` and the official train detection
JSON.

NOT EXERCISED: kept for the case where a licensed BDD100K train copy becomes
available. In round 2 every reachable BDD100K source turned out to be val-only or
labels-free (see docs/road_domain_pilot.md), so the pilot ran on nuImages via
tools/prepare_nuimages_road_subset.py instead. Treat this file as untested until
it has actually produced a subset. It refuses validation images and accepts an optional text file containing
the frozen holdout names as an additional boundary check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

CLASS_MAP = {
    "person": "person", "pedestrian": "person", "rider": "person",
    "bike": "bicycle", "bicycle": "bicycle", "motor": "motorcycle",
    "motorcycle": "motorcycle", "car": "car", "bus": "bus", "truck": "truck",
    "traffic light": "traffic light",
}
CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bdd-root", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True, help="official BDD train JSON")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--holdout-names", type=Path, help="one frozen BDD holdout filename per line")
    return p.parse_args()


def _load_holdout(path: Path | None) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()} if path else set()


def _records(labels_path: Path, holdout: set[str]) -> list[dict]:
    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    records = []
    for item in raw:
        name = Path(item["name"]).name
        if name in holdout:
            raise ValueError(f"holdout image appears in training labels: {name}")
        mapped = []
        for label in item.get("labels", []):
            category = CLASS_MAP.get(label.get("category", ""))
            box = label.get("box2d") or {}
            if category and all(k in box for k in ("x1", "y1", "x2", "y2")):
                mapped.append((category, float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])))
        if mapped:
            records.append({"name": name, "labels": mapped, "attributes": item.get("attributes", {})})
    return records


def _choose(records: list[dict], limit: int, seed: int) -> list[dict]:
    if limit < 1 or limit > len(records):
        raise ValueError(f"limit must be in [1, {len(records)}]")
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    # Round-robin rare classes first, then fill remaining slots randomly.
    buckets = {c: [r for r in shuffled if any(label[0] == c for label in r["labels"])] for c in CLASSES}
    chosen: dict[str, dict] = {}
    index = {c: 0 for c in CLASSES}
    while len(chosen) < limit and any(index[c] < len(buckets[c]) for c in CLASSES):
        progress = False
        for c in sorted(CLASSES, key=lambda x: len(buckets[x])):
            while index[c] < len(buckets[c]) and buckets[c][index[c]]["name"] in chosen:
                index[c] += 1
            if len(chosen) >= limit or index[c] >= len(buckets[c]):
                continue
            record = buckets[c][index[c]]
            chosen[record["name"]] = record
            index[c] += 1
            progress = True
        if not progress:
            break
    return list(chosen.values())


def main() -> None:
    args = _parse_args()
    image_root = args.bdd_root / "images" / "100k" / "train"
    if not image_root.is_dir() or not args.labels.is_file():
        raise FileNotFoundError("expected BDD train images and official train labels")
    holdout = _load_holdout(args.holdout_names)
    records = _records(args.labels, holdout)
    selected = _choose(records, args.limit, args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(selected)
    split_at = max(1, round(len(selected) * 0.9))
    outputs = {"train": selected[:split_at], "val": selected[split_at:]}
    args.output.mkdir(parents=True, exist_ok=True)
    for split, items in outputs.items():
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
        for item in items:
            src = image_root / item["name"]
            if not src.is_file():
                raise FileNotFoundError(src)
            dst = args.output / "images" / split / src.name
            shutil.copy2(src, dst)
            # BDD keyframes are 1280x720; use actual dimensions only if needed.
            lines = []
            for category, x1, y1, x2, y2 in item["labels"]:
                xc, yc = ((x1 + x2) / 2 / 1280, (y1 + y2) / 2 / 720)
                w, h = ((x2 - x1) / 1280, (y2 - y1) / 720)
                if w > 0 and h > 0:
                    lines.append(f"{CLASSES.index(category)} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}")
            (args.output / "labels" / split / f"{src.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = Counter(label[0] for item in selected for label in item["labels"])
    manifest = {"source": str(args.labels), "seed": args.seed, "limit": args.limit, "classes": list(CLASSES), "images": {k: [x["name"] for x in v] for k, v in outputs.items()}, "instance_counts": counts}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"selected": len(selected), "splits": {k: len(v) for k, v in outputs.items()}, "instance_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
