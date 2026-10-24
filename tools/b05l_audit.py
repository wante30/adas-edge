"""B-05L dataset audit: splits, labels, empty negatives, small boxes, BDD50.

The 512x192 training canvases are produced from the top-60% ROI of 1280x720
BDD100K frames without padding, so a canvas pixel is 2.5 original pixels wide.
The audit keeps those two coordinate spaces separate and never treats images
without a BDD label as confirmed hard negatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.b05k_common import size_group  # noqa: E402

DEFAULT_DATA = Path(r"D:\Datasets\adas_edge_training\tl_specialist\data_512x192")
DEFAULT_BDD_IMAGES = Path(r"D:\Datasets\adas_edge_training\b03c_rknn_validation_20261021\board_holdout50")
DEFAULT_BDD_GT = Path(r"D:\Datasets\adas_edge_training\b03d_gt_hf_20261021")
DEFAULT_HOLDOUT_NAMES = Path(r"D:\Datasets\adas_edge_training\bdd_holdout_1000_names.txt")
COLOR_BGR = {"RED": (40, 40, 235), "GREEN": (40, 210, 60), "YELLOW": (30, 220, 240), "UNKNOWN": (220, 220, 220)}


def read_split(root: Path, split: str) -> list[tuple[Path, Path]]:
    images = sorted((root / "images" / split).glob("*"))
    pairs = []
    for image in images:
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        pairs.append((image, root / "labels" / split / f"{image.stem}.txt"))
    return pairs


def parse_label(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        _, cx, cy, bw, bh = map(float, fields[:5])
        x1 = max(0.0, (cx - bw / 2) * width)
        y1 = max(0.0, (cy - bh / 2) * height)
        x2 = min(float(width), (cx + bw / 2) * width)
        y2 = min(float(height), (cy + bh / 2) * height)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({"bbox": [x1, y1, x2, y2], "width_canvas_px": x2 - x1, "height_canvas_px": y2 - y1,
                      "width_original_px": (x2 - x1) * 2.5})
    return boxes


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def draw_sample(image: np.ndarray, boxes: list[dict[str, Any]], title: str) -> np.ndarray:
    canvas = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in box["bbox"])
        color = COLOR_BGR[box.get("color", "UNKNOWN")]
        cv2.rectangle(canvas, (x1, y1), (max(x1 + 1, x2), max(y1 + 1, y2)), color, 1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 18), (20, 24, 28), -1)
    cv2.putText(canvas, title, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def bdd50_audit(images_dir: Path, gt_dir: Path, roi_top: float = 0.60) -> dict[str, Any]:
    rows = []
    color_counts = Counter()
    size_counts = Counter()
    roi_size_counts = Counter()
    total = roi = outside = crossing = 0
    for path in sorted(gt_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        name = str(data["name"])
        image_path = images_dir / f"{name}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        roi_height = int(round(height * roi_top))
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                if obj.get("category") != "traffic light":
                    continue
                b = obj["box2d"]
                x1, y1, x2, y2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
                if x2 <= x1 or y2 <= y1:
                    continue
                total += 1
                color = str(obj.get("attributes", {}).get("trafficLightColor", "none")).lower()
                color_counts[color] += 1
                box_w = x2 - x1
                if y2 <= roi_height:
                    roi += 1
                    size_counts[size_group(box_w)] += 1
                elif y1 >= roi_height:
                    outside += 1
                else:
                    crossing += 1
        rows.append({"name": name, "width": width, "height": height})
    return {
        "images": len(rows),
        "all_gt": total,
        "roi_gt": roi,
        "outside_roi_gt": outside,
        "crossing_roi_gt": crossing,
        "roi_coverage": roi / total if total else 0.0,
        "colors_all": dict(color_counts),
        "roi_size_groups": dict(size_counts),
        "roi_top_fraction": roi_top,
        "image_size": sorted({(row["width"], row["height"]) for row in rows}),
    }


def audit_split(root: Path, split: str, max_samples: int = 12) -> tuple[dict[str, Any], list[np.ndarray]]:
    pairs = read_split(root, split)
    box_count = 0
    empty_images = 0
    missing_labels = 0
    sizes = Counter()
    widths = []
    per_image = []
    hashes = Counter()
    selected: list[tuple[Path, list[dict[str, Any]], str]] = []
    for image_path, label_path in pairs:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        boxes = parse_label(label_path, width, height)
        if not label_path.exists():
            missing_labels += 1
        if not boxes:
            empty_images += 1
        else:
            per_image.append(len(boxes))
        for box in boxes:
            box_count += 1
            sizes[size_group(float(box["width_original_px"]))] += 1
            widths.append(float(box["width_original_px"]))
        hashes[image_sha256(image_path)] += 1
        if len(selected) < max_samples and boxes:
            selected.append((image_path, boxes, f"{split} {len(boxes)} boxes"))
    # Append a few empty canvases and very-small-box canvases for visual audit.
    empties = [(p, [], "EMPTY label (not asserted hard negative)")
               for p, lp in pairs if not parse_label(lp, 512, 192)][:4]
    small = []
    for image_path, label_path in pairs:
        boxes = parse_label(label_path, 512, 192)
        if boxes and min(box["width_original_px"] for box in boxes) < 8:
            small.append((image_path, boxes, "small <8px original-width GT"))
        if len(small) >= 4:
            break
    selected.extend(empties)
    selected.extend(small)
    rng = random.Random(0)
    rng.shuffle(selected)
    sheets = []
    for image_path, boxes, title in selected[:max_samples]:
        image = cv2.imread(str(image_path))
        sheets.append(draw_sample(image, boxes, f"{title} | {image_path.name}"))
    duplicate_hashes = sum(1 for count in hashes.values() if count > 1)
    return {
        "images": len(pairs),
        "labeled_images": len(per_image),
        "empty_label_images": empty_images,
        "missing_label_files": missing_labels,
        "boxes": box_count,
        "boxes_per_labeled_image_mean": float(np.mean(per_image)) if per_image else 0.0,
        "boxes_per_labeled_image_max": max(per_image) if per_image else 0,
        "size_groups_original_width": dict(sizes),
        "width_original_px": {
            "min": min(widths) if widths else None,
            "p50": float(np.percentile(widths, 50)) if widths else None,
            "p95": float(np.percentile(widths, 95)) if widths else None,
            "max": max(widths) if widths else None,
        },
        "exact_duplicate_image_groups": duplicate_hashes,
    }, sheets


def contact_sheet(images: list[np.ndarray], path: Path, *, columns: int = 3) -> None:
    if not images:
        return
    target_w = 512
    target_h = 192
    rows = (len(images) + columns - 1) // columns
    sheet = np.full((rows * (target_h + 8), columns * (target_w + 8), 3), 32, np.uint8)
    for index, image in enumerate(images):
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        row, col = divmod(index, columns)
        y, x = row * (target_h + 8), col * (target_w + 8)
        sheet[y:y + target_h, x:x + target_w] = image
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--bdd-images", type=Path, default=DEFAULT_BDD_IMAGES)
    parser.add_argument("--bdd-gt", type=Path, default=DEFAULT_BDD_GT)
    parser.add_argument("--holdout-names", type=Path, default=DEFAULT_HOLDOUT_NAMES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sheet", type=Path)
    args = parser.parse_args()

    root = args.data
    dataset_yaml = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
    train, train_sheets = audit_split(root, "train")
    val, val_sheets = audit_split(root, "val")
    bdd = bdd50_audit(args.bdd_images, args.bdd_gt)
    train_names = {p.stem for p, _ in read_split(root, "train")}
    val_names = {p.stem for p, _ in read_split(root, "val")}
    bdd_names = {str(json.loads(p.read_text(encoding="utf-8"))["name"]) for p in args.bdd_gt.glob("*.json")}
    holdout_names = set()
    if args.holdout_names.exists():
        holdout_names = {line.strip().rsplit(".", 1)[0] for line in args.holdout_names.read_text(encoding="utf-8").splitlines() if line.strip()}
    result = {
        "dataset_yaml": str((root / "dataset.yaml").resolve()),
        "dataset_declared": dataset_yaml,
        "train": train,
        "val": val,
        "bdd50": bdd,
        "leakage_checks": {
            "bdd50_names_in_train": sorted(bdd_names & train_names),
            "bdd50_names_in_val": sorted(bdd_names & val_names),
            "bdd50_names_not_in_declared_holdout": sorted(bdd_names - holdout_names)[:20] if holdout_names else None,
            "train_val_name_overlap": sorted(train_names & val_names),
            "declared_holdout_names_file": str(args.holdout_names) if args.holdout_names.exists() else None,
        },
        "notes": [
            "Training canvases are 512x192 direct ROI resizes; original width = canvas width * 2.5.",
            "Empty-label images are BDD frames with no traffic-light annotation in the top ROI; they are not counted as proven hard negatives.",
            "BDD50 is a fixed human-GT holdout used only for evaluation; it is excluded from the training split.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    if args.sheet:
        contact_sheet(train_sheets + val_sheets, args.sheet)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "train": train,
        "val": val,
        "bdd50": bdd,
        "leakage_checks": result["leakage_checks"],
    }, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
