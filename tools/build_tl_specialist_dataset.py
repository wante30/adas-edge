"""Build the traffic-light specialist dataset from BDD100K val (non-holdout).

Source: the 9000 BDD100K-100k-val images that are NOT in the frozen 1000-image
holdout (holdout names come from bdd_holdout_1000_names.txt and are excluded
here; the holdout itself is never read by this script beyond exclusion).

Geometry: every image is cropped to the deployment ROI (full width x top 60%,
i.e. 1280x432) and letterboxed onto the specialist canvas (default 512x192),
exactly how the FRONT-only branch will consume frames on the board. Labels are
single-class (0 = traffic light), boxes clipped to the ROI; boxes that lose
more than 70% of their area to the crop are dropped.

Split: deterministic image-level hash split (md5 of filename), no val image
appears in train. The frozen holdout remains acceptance-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def letterbox_bgr(image: np.ndarray, size: tuple[int, int]):
    """Same letterbox contract as tools.profile_rknn_single_stream (BGR in/out)."""
    width, height = size
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    new_w, new_h = round(source_w * scale), round(source_h * scale)
    pad_x = (width - new_w) // 2
    pad_y = (height - new_h) // 2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def split_of(name: str, val_mod: int) -> str:
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return "val" if int(digest[:8], 16) % val_mod == 0 else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdd-root", type=Path, required=True,
                        help="dir containing bdd100k.csv and images/100k/val")
    parser.add_argument("--holdout-names", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi-top", type=float, default=0.60)
    parser.add_argument("--canvas", default="512x192")
    parser.add_argument("--min-area-fraction", type=float, default=0.30,
                        help="drop TL boxes whose visible area after ROI clipping is below this")
    parser.add_argument("--val-mod", type=int, default=18,
                        help="every Nth image by md5 goes to val")
    args = parser.parse_args()

    canvas_w, canvas_h = (int(v) for v in args.canvas.lower().split("x"))
    holdout = {line.strip() for line in args.holdout_names.read_text().splitlines() if line.strip()}
    roi_h = int(round(720 * args.roi_top))

    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    per_image: dict[str, list[tuple[float, float, float, float]]] = {}
    with (args.bdd_root / "bdd100k.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = Path(row["image_path"]).name
            if name in holdout:
                continue
            if row["class_name"] != "traffic light":
                continue
            per_image.setdefault(name, []).append(
                (float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"]))
            )

    counts = Counter()
    box_total = Counter()
    dropped_boxes = 0
    manifest: list[dict] = []

    # All non-holdout images (background frames included, they suppress FPs),
    # TL labels only where they exist.
    all_names: set[str] = set()
    with (args.bdd_root / "bdd100k.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = Path(row["image_path"]).name
            if name not in holdout:
                all_names.add(name)
    for name in sorted(all_names):
        split = split_of(name, args.val_mod)
        source = args.bdd_root / "images" / "100k" / "val" / name
        image = cv2.imread(str(source))
        if image is None:
            counts[f"{split}_unreadable"] += 1
            continue
        crop = image[:roi_h, :, :]
        crop_h, crop_w = crop.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox_bgr(crop, (canvas_w, canvas_h))

        lines = []
        for x1, y1, x2, y2 in per_image.get(name, []):
            # clip to the ROI in source coordinates
            cy1, cy2 = max(0.0, y1), min(float(roi_h), y2)
            if cy2 - cy1 <= 1:
                dropped_boxes += 1
                continue
            visible = ((min(x2, float(crop_w)) - max(x1, 0.0)) * (cy2 - cy1))
            original = (x2 - x1) * (y2 - y1)
            if visible < args.min_area_fraction * original:
                dropped_boxes += 1
                continue
            # source full-frame -> ROI crop -> letterboxed canvas
            bx1 = max(x1, 0.0) * scale + pad_x
            by1 = cy1 * scale + pad_y
            bx2 = min(x2, float(crop_w)) * scale + pad_x
            by2 = cy2 * scale + pad_y
            w = max(0.0, bx2 - bx1)
            h = max(0.0, by2 - by1)
            if w < 1 or h < 1:
                dropped_boxes += 1
                continue
            cx, cy = (bx1 + bx2) / 2 / canvas_w, (by1 + by2) / 2 / canvas_h
            lines.append(f"0 {cx:.6f} {cy:.6f} {w / canvas_w:.6f} {h / canvas_h:.6f}")
            box_total[split] += 1

        stem = Path(name).stem
        (args.output / "images" / split / f"{stem}.jpg").parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output / "images" / split / f"{stem}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (args.output / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[f"{split}_images"] += 1
        manifest.append({"name": name, "split": split, "tl_boxes": len(lines)})

    report = {
        "source": str(args.bdd_root),
        "holdout_names": str(args.holdout_names),
        "roi_top_fraction": args.roi_top,
        "roi_px": [1280, roi_h],
        "canvas": f"{canvas_w}x{canvas_h}",
        "classes": ["traffic light"],
        "candidate_images_with_tl": len(per_image),
        "total_images": len(all_names),
        "dropped_boxes": dropped_boxes,
        "counts": dict(counts),
        "boxes": dict(box_total),
        "split_rule": f"md5(name)[:8] % {args.val_mod} == 0 -> val",
        "holdout_excluded": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dataset_card.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (args.output / "dataset.yaml").write_text(
        f"path: {args.output.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: traffic light\n",
        encoding="utf-8",
    )
    (args.output / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
