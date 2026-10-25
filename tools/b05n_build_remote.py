"""Build the B-05N 10-class dataset on Linux from the downloaded BDD val mirror."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CLASSES = ["person", "rider", "car", "truck", "bus", "train", "motor", "bike", "traffic light", "traffic sign"]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}
LABEL_ALIAS = {"pedestrian": "person", "person": "person", "rider": "rider", "car": "car", "truck": "truck",
               "bus": "bus", "train": "train", "motorcycle": "motor", "motor": "motor", "bicycle": "bike",
               "bike": "bike", "traffic light": "traffic light", "traffic sign": "traffic sign"}
IGNORED = {"other vehicle", "trailer", "other person"}
COLOR_BGR = {"person": (0, 200, 255), "rider": (255, 200, 0), "car": (0, 255, 0), "truck": (0, 128, 255),
             "bus": (255, 0, 255), "train": (128, 0, 255), "motor": (255, 128, 0), "bike": (0, 255, 255),
             "traffic light": (0, 0, 255), "traffic sign": (255, 255, 0)}


def names_from_file(path: Path) -> list[str]:
    return [Path(line.strip()).stem for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert(split: str, names: list[str], samples: dict[str, dict], images: Path, out: Path) -> dict[str, Any]:
    counts, colors, sizes = Counter(), Counter(), Counter()
    missing_images, missing_labels = [], []
    empty = 0
    for name in names:
        sample = samples.get(name)
        if sample is None:
            missing_labels.append(name)
            continue
        src = images / f"{name}.jpg"
        if not src.exists():
            missing_images.append(name)
            continue
        link(src, out / "images" / split / f"{name}.jpg")
        lines = []
        for det in sample.get("detections", {}).get("detections", []):
            mapped = LABEL_ALIAS.get(str(det.get("label", "")))
            if mapped is None:
                continue
            x, y, w, h = (float(v) for v in det.get("bounding_box", [0, 0, 0, 0])[:4])
            x1, y1, x2, y2 = max(0.0, min(1.0, x)), max(0.0, min(1.0, y)), max(0.0, min(1.0, x + w)), max(0.0, min(1.0, y + h))
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(f"{CLASS_ID[mapped]} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}\n")
            counts[mapped] += 1
            if mapped == "traffic light":
                width_px = (x2 - x1) * 1280.0
                sizes["<8" if width_px < 8 else "8-16" if width_px < 16 else "16-32" if width_px < 32 else ">32"] += 1
                colors[str(det.get("trafficLightColor", "NA"))] += 1
        label_path = out / "labels" / split / f"{name}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("".join(lines), encoding="utf-8")
        if not lines:
            empty += 1
    return {"requested": len(names), "written": len(names) - len(missing_images) - len(missing_labels),
            "missing_image_count": len(missing_images), "missing_label_count": len(missing_labels),
            "missing_images": missing_images[:20], "missing_labels": missing_labels[:20],
            "empty_after_filter": empty, "class_counts": dict(counts),
            "traffic_light_size_groups": dict(sizes), "traffic_light_colors": dict(colors)}


def draw(sample: dict, image: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    h, w = image.shape[:2]
    for det in sample.get("detections", {}).get("detections", []):
        mapped = LABEL_ALIAS.get(str(det.get("label", "")))
        if mapped is None:
            continue
        x, y, bw, bh = (float(v) for v in det.get("bounding_box", [0, 0, 0, 0])[:4])
        x1, y1, x2, y2 = int(x*w), int(y*h), int((x+bw)*w), int((y+bh)*h)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_BGR[mapped], 1)
    return canvas


def contact_sheet(images: list[np.ndarray], path: Path) -> None:
    if not images:
        return
    width, height, cols = 640, 360, 2
    rows = (len(images) + cols - 1) // cols
    sheet = np.full((rows*height, cols*(width+6), 3), 28, np.uint8)
    for i, image in enumerate(images):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        sheet[r*height:(r+1)*height, c*(width+6):c*(width+6)+width] = image
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--names-dir", type=Path, required=True)
    ap.add_argument("--bdd50-gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--sheet", type=Path)
    args = ap.parse_args()
    samples_list = json.loads(args.samples.read_text(encoding="utf-8"))["samples"]
    samples = {Path(row["filepath"]).stem: row for row in samples_list}
    train = names_from_file(args.names_dir / "train_names.txt")
    val = names_from_file(args.names_dir / "val_names.txt")
    hold = names_from_file(args.names_dir / "holdout_names.txt")
    bdd50 = sorted(json.loads(p.read_text(encoding="utf-8"))["name"] for p in args.bdd50_gt.glob("*.json"))
    valset, holdset = set(val), set(hold)
    train = [n for n in train if n not in valset and n not in holdset]
    val = [n for n in val if n not in set(train) and n not in holdset]
    hold = [n for n in hold if n not in set(train) and n not in set(val)]
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"train": convert("train", train, samples, args.images, args.out),
              "val": convert("val", val, samples, args.images, args.out),
              "test": convert("test", hold, samples, args.images, args.out)}
    yaml = args.out / "data.yaml"
    yaml.write_text(f"path: {args.out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" +
                    "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES)), encoding="utf-8")
    manifest = {"source_samples": str(args.samples), "source_samples_sha256": sha256(args.samples),
                "images": str(args.images), "classes": CLASSES, "ignored": sorted(IGNORED),
                "label_aliases": LABEL_ALIAS, "input_canvas": "full 16:9, imgsz=512 -> 512x288",
                "splits": result, "bdd50_in_test": sorted(set(bdd50) & set(hold)),
                "bdd50_in_train": sorted(set(bdd50) & set(train)), "bdd50_in_val": sorted(set(bdd50) & set(val)),
                "dataset_yaml": str(yaml), "note": "BDD100K validation mirror, 9k train + 1k holdout; not the 70k train split."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    rng = random.Random(0)
    chosen = [samples[n] for n in rng.sample([n for n in train if n in samples], min(12, len(train)))]
    images = []
    for sample in chosen:
        name = Path(sample["filepath"]).stem
        image = cv2.imread(str(args.images / f"{name}.jpg"))
        if image is not None:
            images.append(draw(sample, image))
    if args.sheet:
        contact_sheet(images, args.sheet)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "missing_images"} for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
