"""B-05L data preparation: validate the shared split and build an optional domain set.

The primary comparison uses exactly the 8513/487 canvases that trained the
production YOLO11n Specialist.  The 15 independently reviewed competition-video
images are converted into the same top-60% 512x192 canvas and kept as a separate
domain-adaptation set; they are not silently mixed into the primary comparison.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA = Path(r"D:\Datasets\adas_edge_training\tl_specialist\data_512x192")
DEFAULT_PSEUDO = Path(r"D:\Datasets\adas_edge_training\b05i_clean_20261024\pseudo_A")
DEFAULT_OUT = Path(r"D:\Datasets\adas_edge_training\b05l_data")


def count_images(path: Path) -> int:
    return sum(1 for p in path.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def validate_canvas(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "val"):
        images = sorted((root / "images" / split).glob("*"))
        labels = sorted((root / "labels" / split).glob("*.txt"))
        bad = []
        empty = 0
        boxes = 0
        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                bad.append({"image": str(image_path), "reason": "unreadable"})
                continue
            if image.shape[:2] != (192, 512):
                bad.append({"image": str(image_path), "reason": f"shape={image.shape[:2]}"})
            label = root / "labels" / split / f"{image_path.stem}.txt"
            if not label.exists():
                bad.append({"image": str(image_path), "reason": "missing label"})
                continue
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty += 1
            for line in lines:
                fields = line.split()
                if len(fields) != 5:
                    bad.append({"label": str(label), "reason": "field count"})
                    continue
                _, cx, cy, bw, bh = map(float, fields)
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    bad.append({"label": str(label), "reason": "out of range"})
                boxes += 1
        result[split] = {
            "images": len(images), "labels": len(labels), "empty_label_images": empty,
            "boxes": boxes, "invalid": bad[:20], "invalid_count": len(bad),
        }
    return result


def dhash(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    return (image[:, 1:] > image[:, :-1]).flatten().astype(np.uint8)


def clean_validation(root: Path, out: Path, distance: int) -> dict[str, Any]:
    """Remove validation frames whose dHash is nearly identical to a train frame.

    BDD100K names do not expose source-video IDs.  dHash <= 3 catches obvious
    adjacent/near-identical frames conservatively without claiming a full
    video-level split.  The excluded images are not moved into train.
    """
    train = [(p.stem, dhash(p)) for p in sorted((root / "images" / "train").glob("*.jpg"))]
    train_stack = np.stack([value for _, value in train])
    excluded = []
    kept = []
    for image_path in sorted((root / "images" / "val").glob("*.jpg")):
        value = dhash(image_path)
        distances = (train_stack ^ value).sum(axis=1)
        nearest = int(distances.min())
        nearest_name = train[int(distances.argmin())][0]
        if nearest <= distance:
            excluded.append({"image": image_path.name, "nearest_train": nearest_name, "dhash_distance": nearest})
        else:
            kept.append(image_path)
    for split in ("images", "labels"):
        (out / "val_clean" / split).mkdir(parents=True, exist_ok=True)
    for image_path in kept:
        shutil.copy2(image_path, out / "val_clean" / "images" / image_path.name)
        label = root / "labels" / "val" / f"{image_path.stem}.txt"
        shutil.copy2(label, out / "val_clean" / "labels" / label.name)
    return {"threshold": distance, "val_images": len(kept) + len(excluded), "kept": len(kept),
            "excluded": excluded, "rule": "exclude val images with dHash distance <= threshold to any train image"}


def convert_pseudo(pseudo: Path, out: Path) -> dict[str, Any]:
    rows = []
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image_path in sorted((pseudo / "images" / split).glob("*.jpg")):
            label_path = pseudo / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            roi_height = int(round(height * 0.60))
            canvas = cv2.resize(image[:roi_height], (512, 192), interpolation=cv2.INTER_AREA)
            lines = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) < 5:
                    continue
                _, cx, cy, bw, bh = map(float, fields[:5])
                x1, x2 = (cx - bw / 2) * width, (cx + bw / 2) * width
                y1, y2 = (cy - bh / 2) * height, (cy + bh / 2) * height
                y1, y2 = max(0.0, min(float(roi_height), y1)), max(0.0, min(float(roi_height), y2))
                x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
                if x2 <= x1 or y2 <= y1:
                    continue
                lines.append(
                    f"0 {(x1+x2)/2/width:.6f} {(y1+y2)/2/roi_height:.6f} "
                    f"{(x2-x1)/width:.6f} {(y2-y1)/roi_height:.6f}\n"
                )
            shutil.copy2(image_path, out / "images" / split / image_path.name)
            (out / "labels" / split / f"{image_path.stem}.txt").write_text("".join(lines), encoding="utf-8")
            rows.append({"image": image_path.name, "split": split, "boxes": len(lines),
                         "source_size": [width, height]})
    return {"images": len(rows), "splits": {split: sum(r["split"] == split for r in rows) for split in ("train", "val", "test")},
            "boxes": sum(r["boxes"] for r in rows), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--pseudo", type=Path, default=DEFAULT_PSEUDO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean-val-distance", type=int, default=3,
                        help="exclude val images with dHash distance <= this to any train image")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    validation = validate_canvas(args.data)
    clean_val = clean_validation(args.data, args.out, args.clean_val_distance)
    domain = convert_pseudo(args.pseudo, args.out / "domain")
    primary_yaml = args.out / "primary.yaml"
    primary_yaml.write_text(
        f"path: {args.data}\ntrain: images/train\nval: images/val\nnames:\n  0: traffic light\n",
        encoding="utf-8",
    )
    clean_yaml = args.out / "primary_clean.yaml"
    clean_yaml.write_text(
        f"path: {args.data}\ntrain: images/train\nval: {args.out / 'val_clean' / 'images'}\nnames:\n  0: traffic light\n",
        encoding="utf-8",
    )
    domain_yaml = args.out / "domain_finetune.yaml"
    domain_yaml.write_text(
        f"path: {args.out / 'domain'}\ntrain: images/train\nval: images/val\nnames:\n  0: traffic light\n",
        encoding="utf-8",
    )
    combined_yaml = args.out / "primary_plus_domain.yaml"
    combined_yaml.write_text(
        "path: " + str(args.data) + "\n"
        "train:\n"
        f"  - {args.data / 'images' / 'train'}\n"
        f"  - {args.out / 'domain' / 'images' / 'train'}\n"
        "val:\n"
        f"  - {args.out / 'val_clean' / 'images'}\n"
        f"  - {args.out / 'domain' / 'images' / 'val'}\n"
        "names:\n  0: traffic light\n",
        encoding="utf-8",
    )
    result = {
        "primary_data": str(args.data.resolve()),
        "validation": validation,
        "clean_validation": clean_val,
        "domain": {k: v for k, v in domain.items() if k != "rows"},
        "yaml": {
            "primary": str(primary_yaml),
            "primary_clean": str(clean_yaml),
            "domain": str(domain_yaml),
            "primary_plus_domain": str(combined_yaml),
        },
        "fairness_note": (
            "Primary comparison uses the same 8513/487 split that trained the production YOLO11n. "
            "The 15 video-domain images are a separate adaptation experiment, not part of the primary comparison."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
