"""B-05N build a full-frame 512x288 ten-class BDD dataset.

Source: BDD100K validation mirror with full-frame JPEGs and FiftyOne-style
detections (dgural/bdd100k).  The split reuses the exact B-05L train/clean-val
names and the B-03D holdout-1000 names, so BDD50 is never trained on.
"""
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
CLASS_ID = {name: index for index, name in enumerate(CLASSES)}
LABEL_ALIAS = {
    "pedestrian": "person",
    "person": "person",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motorcycle": "motor",
    "motor": "motor",
    "bicycle": "bike",
    "bike": "bike",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
}
IGNORED = {"other vehicle", "trailer", "other person"}
COLOR_BGR = {
    "person": (0, 200, 255), "rider": (255, 200, 0), "car": (0, 255, 0), "truck": (0, 128, 255),
    "bus": (255, 0, 255), "train": (128, 0, 255), "motor": (255, 128, 0), "bike": (0, 255, 255),
    "traffic light": (0, 0, 255), "traffic sign": (255, 255, 0),
}


def names_from_dir(path: Path) -> list[str]:
    return sorted(p.stem for p in path.glob("*.jpg"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_split(split: str, names: list[str], samples: dict[str, dict], images_dir: Path, out: Path) -> dict[str, Any]:
    counts = Counter()
    colors = Counter()
    sizes = Counter()
    empty = 0
    missing_images = []
    missing_labels = []
    out_images = out / "images" / split
    out_labels = out / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    for name in names:
        sample = samples.get(name)
        if sample is None:
            missing_labels.append(name)
            continue
        src = images_dir / f"{name}.jpg"
        if not src.exists():
            missing_images.append(name)
            continue
        link_or_copy(src, out_images / f"{name}.jpg")
        lines = []
        for det in sample.get("detections", {}).get("detections", []):
            original = str(det.get("label", ""))
            mapped = LABEL_ALIAS.get(original)
            if mapped is None:
                continue
            box = det.get("bounding_box", [0, 0, 0, 0])
            x, y, w, h = (float(v) for v in box[:4])
            x1, y1, x2, y2 = max(0.0, min(1.0, x)), max(0.0, min(1.0, y)), max(0.0, min(1.0, x + w)), max(0.0, min(1.0, y + h))
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(f"{CLASS_ID[mapped]} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}\n")
            counts[mapped] += 1
            if mapped == "traffic light":
                width_px = (x2 - x1) * 1280.0
                sizes["<8" if width_px < 8 else "8-16" if width_px < 16 else "16-32" if width_px < 32 else ">32"] += 1
                colors[str(det.get("trafficLightColor", "NA"))] += 1
        (out_labels / f"{name}.txt").write_text("".join(lines), encoding="utf-8")
        if not lines:
            empty += 1
    return {
        "requested": len(names), "written": len(names) - len(missing_images) - len(missing_labels),
        "missing_images": missing_images[:50], "missing_labels": missing_labels[:50],
        "missing_image_count": len(missing_images), "missing_label_count": len(missing_labels),
        "empty_after_filter": empty, "class_counts": dict(counts),
        "traffic_light_size_groups": dict(sizes), "traffic_light_colors": dict(colors),
    }


def draw_sample(sample: dict, image: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    for det in sample.get("detections", {}).get("detections", []):
        mapped = LABEL_ALIAS.get(str(det.get("label", "")))
        if mapped is None:
            continue
        x, y, w, h = (float(v) for v in det.get("bounding_box", [0, 0, 0, 0])[:4])
        x1, y1, x2, y2 = int(x * image.shape[1]), int(y * image.shape[0]), int((x + w) * image.shape[1]), int((y + h) * image.shape[0])
        color = COLOR_BGR[mapped]
        cv2.rectangle(canvas, (x1, y1), (max(x1 + 1, x2), max(y1 + 1, y2)), color, 1)
        cv2.putText(canvas, mapped, (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return canvas


def contact_sheet(images: list[np.ndarray], path: Path, columns: int = 2) -> None:
    if not images:
        return
    width, height = 640, 360
    rows = (len(images) + columns - 1) // columns
    sheet = np.full((rows * height, columns * (width + 6), 3), 28, np.uint8)
    for index, image in enumerate(images):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        row, col = divmod(index, columns)
        sheet[row * height:(row + 1) * height, col * (width + 6):col * (width + 6) + width] = image
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-json", type=Path, default=Path(r"D:\Datasets\bdd_val_samples.json"))
    parser.add_argument("--images-dir", type=Path, default=Path(r"D:\Datasets\bdd100k_val_hf\data"))
    parser.add_argument("--tl-data", type=Path, default=Path(r"D:\Datasets\adas_edge_training\tl_specialist\data_512x192"))
    parser.add_argument("--clean-val", type=Path, default=Path(r"D:\Datasets\adas_edge_training\b05l_data\val_clean\images"))
    parser.add_argument("--holdout-names", type=Path, default=Path(r"D:\Datasets\adas_edge_training\bdd_holdout_1000_names.txt"))
    parser.add_argument("--bdd50-gt", type=Path, default=Path(r"D:\Datasets\adas_edge_training\b03d_gt_hf_20261021"))
    parser.add_argument("--out", type=Path, default=Path(r"D:\Datasets\adas_edge_training\b05n_bdd_multiclass"))
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--sheet", type=Path)
    args = parser.parse_args()

    samples_list = json.loads(args.samples_json.read_text(encoding="utf-8"))["samples"]
    samples = {Path(row["filepath"]).stem: row for row in samples_list}
    train_names = names_from_dir(args.tl_data / "images" / "train")
    val_names = names_from_dir(args.clean_val)
    holdout_names = []
    for line in args.holdout_names.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            holdout_names.append(Path(line).stem)
    bdd50_names = sorted(json.loads(p.read_text(encoding="utf-8"))["name"] for p in args.bdd50_gt.glob("*.json"))
    # Remove exact overlap between splits defensively.
    val_set, holdout_set = set(val_names), set(holdout_names)
    train_names = [name for name in train_names if name not in val_set and name not in holdout_set]
    val_names = [name for name in val_names if name not in set(train_names) and name not in holdout_set]
    holdout_names = [name for name in holdout_names if name not in set(train_names) and name not in set(val_names)]

    args.out.mkdir(parents=True, exist_ok=True)
    train = convert_split("train", train_names, samples, args.images_dir, args.out)
    val = convert_split("val", val_names, samples, args.images_dir, args.out)
    test = convert_split("test", holdout_names, samples, args.images_dir, args.out)
    yaml = args.out / "data.yaml"
    yaml.write_text(
        f"path: {args.out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASSES)),
        encoding="utf-8",
    )
    manifest = {
        "source_samples_json": str(args.samples_json.resolve()),
        "source_samples_sha256": sha256(args.samples_json),
        "images_dir": str(args.images_dir.resolve()),
        "classes": CLASSES,
        "ignored_labels": sorted(IGNORED),
        "label_aliases": LABEL_ALIAS,
        "input_canvas": "full 16:9 frame, training imgsz=512 (512x288 with rect=True)",
        "splits": {"train": train, "val": val, "test": test},
        "bdd50_names": bdd50_names,
        "bdd50_in_test": sorted(set(bdd50_names) & set(holdout_names)),
        "bdd50_in_train": sorted(set(bdd50_names) & set(train_names)),
        "bdd50_in_val": sorted(set(bdd50_names) & set(val_names)),
        "dataset_yaml": str(yaml),
        "note": "10k BDD100K validation images; train/val reuse B-05L names; test is the 1000-name holdout. Not the 70k BDD train split.",
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    rng = random.Random(0)
    selected = [samples[name] for name in rng.sample([n for n in train_names if n in samples], min(12, len(train_names)))]
    images = []
    for sample in selected:
        name = Path(sample["filepath"]).stem
        image = cv2.imread(str(args.images_dir / f"{name}.jpg"))
        if image is not None:
            images.append(draw_sample(sample, image))
    if args.sheet:
        contact_sheet(images, args.sheet)
    print(json.dumps({"train": train, "val": val, "test": test, "yaml": str(yaml),
                      "bdd50_in_test": len(manifest["bdd50_in_test"])}, indent=2))


if __name__ == "__main__":
    main()
