"""Build stage-A labels: lane/style mask, drivable mask and YOLO detection labels.

Input : BDD100K per-image JSON (official bdd100k_labels.zip)
        + images/100k/{train,val}
Output: label_mtl/{lane,drivable}/{split}/*.png, label_mtl/det/{split}/*.txt
        + label_mtl/index_{split}.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtl.label_render import (  # noqa: E402
    DRIVABLE_BG, LANE_BG, det_boxes, render_drivable, render_lane,
)

W, H = 512, 288
LANE_WIDTH = 4


def load_frame(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        frame = data[0] if data else {}
        return frame.get("objects") or frame.get("labels") or [], frame.get("attributes") or {}
    frames = data.get("frames") or []
    if frames:
        frame = frames[0]
        attrs = data.get("attributes") or frame.get("attributes") or {}
        return frame.get("objects") or frame.get("labels") or [], attrs
    return data.get("objects") or data.get("labels") or [], data.get("attributes") or {}


def process_one(args):
    sid, json_path, out_root, split, img_size = args
    w, h = img_size
    objects, attrs = load_frame(Path(json_path))

    lane = Image.new("L", (w, h), LANE_BG)
    drv = Image.new("L", (w, h), DRIVABLE_BG)
    n_lane = render_lane(ImageDraw.Draw(lane), objects, w / 1280.0, h / 720.0, LANE_WIDTH)
    n_drv = render_drivable(ImageDraw.Draw(drv), objects, w / 1280.0, h / 720.0)

    lane_arr = np.asarray(lane)
    drv_arr = np.asarray(drv)
    lane_px = int((lane_arr != LANE_BG).sum())
    cross_px = int(((lane_arr != LANE_BG) & ((lane_arr & 7) == 0)).sum())
    has_lane = n_lane > 0 and lane_px > 0
    has_drv = n_drv > 0 and int((drv_arr != DRIVABLE_BG).sum()) > 0

    lane_out = out_root / "label_mtl" / "lane" / split / f"{sid}.png"
    drv_out = out_root / "label_mtl" / "drivable" / split / f"{sid}.png"
    det_out = out_root / "label_mtl" / "det" / split / f"{sid}.txt"
    if has_lane:
        lane.save(lane_out)
    if has_drv:
        drv.save(drv_out)

    boxes = det_boxes(objects)
    if boxes:
        with open(det_out, "w") as f:
            for cls, x1, y1, x2, y2 in boxes:
                cx = (x1 + x2) / 2 / 1280.0
                cy = (y1 + y2) / 2 / 720.0
                bw = (x2 - x1) / 1280.0
                bh = (y2 - y1) / 720.0
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    return {
        "id": sid,
        "lane": bool(has_lane),
        "drivable": bool(has_drv),
        "boxes": len(boxes),
        "lane_px": lane_px,
        "crosswalk_px": cross_px,
        "timeofday": attrs.get("timeofday", "undefined"),
        "weather": attrs.get("weather", "undefined"),
        "scene": attrs.get("scene", "undefined"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/bdd100k")
    ap.add_argument("--labels", default="labels/100k")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: limit images per split")
    args = ap.parse_args()

    root = Path(args.root)
    for split in args.splits.split(","):
        label_dir = root / args.labels / split
        jsons = sorted(label_dir.glob("*.json"))
        if args.limit:
            jsons = jsons[: args.limit]
        if not jsons:
            raise SystemExit(f"no json under {label_dir}")

        for sub in ("lane", "drivable", "det"):
            (root / "label_mtl" / sub / split).mkdir(parents=True, exist_ok=True)

        tasks = [(p.stem, str(p), root, split, (W, H)) for p in jsons]
        rows = []
        with Pool(args.workers) as pool:
            for i, rec in enumerate(pool.imap_unordered(process_one, tasks, chunksize=8), 1):
                rows.append(rec)
                if i % 2000 == 0:
                    print(f"[{split}] {i}/{len(tasks)}", flush=True)

        rows.sort(key=lambda r: r["id"])
        with open(root / "label_mtl" / f"index_{split}.json", "w") as f:
            json.dump(rows, f)

        n = len(rows)
        n_lane = sum(r["lane"] for r in rows)
        n_cross = sum(r["crosswalk_px"] > 0 for r in rows)
        n_drv = sum(r["drivable"] for r in rows)
        print(f"[{split}] images={n} lane={n_lane} crosswalk={n_cross} drivable={n_drv}")

    # make a flat image view so the dataset can open {id}.jpg directly
    for split in args.splits.split(","):
        src = root / "images" / "100k" / split
        dst = root / "images" / "flat" / split
        if not src.exists():
            print(f"[{split}] images missing at {src}, skip flat view")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        if not any(dst.iterdir()):
            for p in src.rglob("*.jpg"):
                link = dst / p.name
                if not link.exists():
                    os.symlink(p.resolve(), link)
        print(f"[{split}] flat images: {len(list(dst.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
