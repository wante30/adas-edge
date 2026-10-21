"""Read-only audit for existing YOLO-format image datasets and BDD label JSON."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _class_names(dataset_yaml: Path) -> dict[int, str]:
    text = dataset_yaml.read_text(encoding="utf-8")
    match = re.search(r"(?ms)^names:\s*\n((?:^[ \t].*\n?)*)", text)
    if not match:
        raise ValueError(f"could not parse class names from {dataset_yaml}")
    names = {
        int(class_id): name.strip().strip("\"'")
        for class_id, name in re.findall(r"(?m)^\s*(\d+):\s*(.*?)\s*$", match.group(1))
    }
    if not names:
        raise ValueError(f"no indexed class names found in {dataset_yaml}")
    return names


def _sequence_id(stem: str) -> str:
    return stem.rsplit("-", 1)[0] if "-" in stem else stem


def audit_dataset(root: Path, dataset_yaml: Path, raw_labels_json: Path | None = None) -> dict:
    names = _class_names(dataset_yaml)
    splits: dict[str, dict] = {}
    class_instances: Counter[str] = Counter()
    class_images: Counter[str] = Counter()
    malformed_rows: list[str] = []
    duplicate_boxes = 0
    missing_label_files = 0
    sequences_by_split: dict[str, set[str]] = {}

    for image_dir in sorted((root / "images").glob("*")):
        if not image_dir.is_dir():
            continue
        split = image_dir.name
        label_dir = root / "labels" / split
        image_paths = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        readable = 0
        dimensions: Counter[str] = Counter()
        sequences_by_split[split] = {_sequence_id(path.stem) for path in image_paths}
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                readable += 1
                dimensions[f"{image.shape[1]}x{image.shape[0]}"] += 1
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                missing_label_files += 1
                continue
            seen: set[tuple] = set()
            image_classes: set[str] = set()
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if not fields:
                    continue
                try:
                    class_id = int(fields[0])
                    coords = tuple(float(value) for value in fields[1:])
                except ValueError:
                    malformed_rows.append(f"{label_path}:{line_number}")
                    continue
                if len(coords) != 4 or class_id not in names or any(value < 0 or value > 1 for value in coords):
                    malformed_rows.append(f"{label_path}:{line_number}")
                    continue
                class_name = names[class_id]
                class_instances[class_name] += 1
                image_classes.add(class_name)
                box_key = (class_id, *coords)
                if box_key in seen:
                    duplicate_boxes += 1
                seen.add(box_key)
            class_images.update(image_classes)
        splits[split] = {
            "images": len(image_paths),
            "readable_images": readable,
            "dimensions": dict(dimensions),
            "label_files": sum(1 for path in label_dir.rglob("*.txt")) if label_dir.exists() else 0,
            "sequences": len(sequences_by_split[split]),
        }

    split_names = list(sequences_by_split)
    shared_sequences = set.intersection(*(sequences_by_split[name] for name in split_names)) if len(split_names) > 1 else set()
    result = {
        "dataset_root": str(root),
        "splits": splits,
        "total_images": sum(item["images"] for item in splits.values()),
        "total_readable_images": sum(item["readable_images"] for item in splits.values()),
        "classes": {
            name: {"instances": class_instances[name], "images": class_images[name]}
            for name in names.values()
        },
        "missing_label_files": missing_label_files,
        "malformed_rows": malformed_rows,
        "exact_duplicate_boxes": duplicate_boxes,
        "shared_sequence_ids_across_splits": len(shared_sequences),
        "shared_sequence_examples": sorted(shared_sequences)[:20],
    }

    if raw_labels_json is not None:
        raw = json.loads(raw_labels_json.read_text(encoding="utf-8"))
        colors: Counter[str] = Counter()
        traffic_light_labels = 0
        labels_with_color = 0
        for frame in raw:
            for label in frame.get("labels", []):
                if label.get("category") != "traffic light":
                    continue
                traffic_light_labels += 1
                attributes = label.get("attributes") or {}
                if "trafficLightColor" in attributes:
                    labels_with_color += 1
                    colors[str(attributes["trafficLightColor"])] += 1
        result["raw_json_traffic_light_color"] = {
            "traffic_light_labels": traffic_light_labels,
            "labels_with_trafficLightColor": labels_with_color,
            "values": dict(colors),
        }
    else:
        result["raw_json_traffic_light_color"] = "not_checked_no_original_json_supplied"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="root containing images/<split> and labels/<split>")
    parser.add_argument("--dataset-yaml", type=Path, required=True, help="YOLO dataset YAML with indexed names")
    parser.add_argument("--raw-labels-json", type=Path, help="optional original BDD detection labels JSON")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    result = audit_dataset(args.dataset_root, args.dataset_yaml, args.raw_labels_json)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
