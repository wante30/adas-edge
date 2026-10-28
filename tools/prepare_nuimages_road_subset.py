"""Build a balanced road-domain YOLO subset from nuImages keyframe boxes.

Data source: ``finedet/nuimages`` on the Hugging Face Hub - the Motional
nuImages v1.0 keyframe camera images with their 2D box annotations converted to
COCO ``[x, y, w, h]`` pixel boxes (CC BY-NC-SA 4.0, see the dataset card).

Why nuImages: BDD100K's official download portal and the ``dl.cv.ethz.ch``
mirror were both unreachable from this machine and no BDD100K *train* image +
box mirror exists on the Hub, while nuImages ships front/back keyframes with
boxes in a single parquet stream. The consequence is recorded honestly: nuImages
has no ``traffic light`` category, so that head is left untouched by this pilot.

Two label modes:

``coco80``   (default) Keep the pretrained 80-class head. Labels carry the
             original COCO class ids, so ``yolo11n.pt`` fine-tunes without any
             head surgery and the result stays directly comparable to BASE-320
             and BASE-416.
``compact7`` Emit a dense 7-class layout for later RKNN deployment. The
             COCO -> 7 transfer itself lives in ``tools/transfer_coco_head.py``.

The frozen BDD100K selection holdout is never read here; ``--holdout-names``
accepts its filename list and makes the script abort if a collision ever appears.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

SHARD_URL = "https://huggingface.co/datasets/finedet/nuimages/resolve/main/data/train-{index:05d}-of-00027.parquet"
USER_AGENT = {"User-Agent": "Mozilla/5.0"}
CAMERA_RE = re.compile(r"__(CAM_[A-Z]+)__")

# nuImages' 23 native foreground categories collapsed onto the seven ADAS
# classes. Categories 8-12 and 17-19 (barrier, traffic cone, trailer, ...) are
# intentionally dropped: they are not part of the competition taxonomy.
NUIMAGES_TO_TARGET: dict[int, str] = {
    1: "person", 2: "person", 3: "person", 4: "person", 5: "person", 6: "person", 7: "person",
    13: "bicycle",
    16: "car",
    20: "motorcycle",
    14: "bus", 15: "bus",
    22: "truck",
}

TARGET_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light")

# Original COCO class ids, used so the pretrained 80-class head is preserved.
COCO_IDS: dict[str, int] = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7, "traffic light": 9,
}
COMPACT_IDS: dict[str, int] = {name: index for index, name in enumerate(TARGET_CLASSES)}

COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

# Smallest accepted box edge in pixels; nuImages has sub-pixel boxes that would
# become training noise once letterboxed down to 512x288.
MIN_BOX_EDGE = 2.0


@dataclass
class Record:
    name: str
    labels: list[tuple[int, float, float, float, float]]  # class index, normalised cx, cy, w, h
    source: str = ""
    score: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def classes(self) -> set[int]:
        return {label[0] for label in self.labels}


def class_index(target: str, mode: str) -> int:
    table = COCO_IDS if mode == "coco80" else COMPACT_IDS
    return table[target]


def camera_of(path: str) -> str:
    match = CAMERA_RE.search(path or "")
    return match.group(1) if match else "unknown"


def clean_boxes(
    boxes: list[list[float]], categories: list[int], width: float, height: float, mode: str
) -> list[tuple[int, float, float, float, float]]:
    """Clip pixel boxes to the frame and return normalised YOLO labels.

    Boxes are dropped (never silently shrunk) when they do not map onto one of
    the seven target classes, when they fall outside the frame, or when either
    remaining edge is below :data:`MIN_BOX_EDGE`.
    """
    labels: list[tuple[int, float, float, float, float]] = []
    if width <= 0 or height <= 0:
        return labels
    for box, category in zip(boxes, categories):
        target = NUIMAGES_TO_TARGET.get(int(category))
        if target is None or len(box) != 4:
            continue
        x, y, w, h = (float(value) for value in box)
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(width), x + w), min(float(height), y + h)
        if x2 - x1 < MIN_BOX_EDGE or y2 - y1 < MIN_BOX_EDGE:
            continue
        labels.append(
            (
                class_index(target, mode),
                round((x1 + x2) / 2 / width, 6),
                round((y1 + y2) / 2 / height, 6),
                round((x2 - x1) / width, 6),
                round((y2 - y1) / height, 6),
            )
        )
    return labels


def select_balanced(
    records: list[Record],
    limit: int,
    seed: int,
    min_instances: dict[int, int],
    rarity_order: list[int],
    monotone_cap: int,
) -> list[Record]:
    """Pick ``limit`` records, rarest-class-first, with a cap on car/person-only frames.

    Raw scene statistics are hopelessly skewed (nuImages has ~25 cars per
    bicycle), which is exactly the "person 9000 / bicycle 200" failure the pilot
    is meant to avoid. Phase A walks classes from rarest to most common and
    spends slots on frames that carry the rare classes until every minimum is
    met; phase B fills the remainder, but frames whose classes all live in
    ``monotone_cap``-exempt common classes are charged against a hard share.
    """
    if limit < 1 or limit > len(records):
        raise ValueError(f"limit must be in [1, {len(records)}]")
    frequency = collections.Counter()
    for record in records:
        for index in record.classes:
            frequency[index] += 1
    for record in records:
        record.score = sum(1.0 / max(frequency[index], 1) for index in record.classes)

    by_class: dict[int, list[int]] = {index: [] for index in rarity_order}
    for position, record in enumerate(records):
        for index in record.classes:
            if index in by_class:
                by_class[index].append(position)
    for positions in by_class.values():
        positions.sort(key=lambda position: (-records[position].score, records[position].name))

    chosen: list[Record] = []
    taken: set[int] = set()
    counts: collections.Counter[int] = collections.Counter()

    for index in rarity_order:  # phase A: satisfy the rare-class floors
        wanted = min_instances.get(index, 0)
        for position in by_class[index]:
            if counts[index] >= wanted or len(chosen) >= limit:
                break
            if position in taken:
                continue
            record = records[position]
            taken.add(position)
            chosen.append(record)
            for label in record.classes:
                counts[label] += 1

    exempt = {index for index in rarity_order if min_instances.get(index, 0) > 0}
    common_only = [position for position, record in enumerate(records) if not (record.classes - exempt)]
    others = [position for position in range(len(records)) if position not in set(common_only)]
    random.Random(seed).shuffle(others)
    random.Random(seed + 1).shuffle(common_only)
    common_budget = max(0, int((limit - len(chosen)) * monotone_cap))

    for position in others + common_only[:common_budget]:
        if len(chosen) >= limit:
            break
        if position in taken:
            continue
        taken.add(position)
        chosen.append(records[position])
    return chosen


def download_shard(index: int, destination: Path) -> Path:
    request = urllib.request.Request(SHARD_URL.format(index=index), headers=USER_AGENT)
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 20)
    return destination


def read_shard(path: Path, mode: str, want_multi_view: bool) -> list[Record]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    records: list[Record] = []
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=["image", "width", "height", "objects"])
        images = table.column("image").to_pylist()
        widths = table.column("width").to_pylist()
        heights = table.column("height").to_pylist()
        objects = table.column("objects").to_pylist()
        for image, width, height, obj in zip(images, widths, heights, objects):
            path_text = image.get("path") or ""
            camera = camera_of(path_text)
            if not want_multi_view and camera not in ("CAM_FRONT", "unknown"):
                continue
            labels = clean_boxes(obj.get("bbox") or [], obj.get("category") or [], width, height, mode)
            if not labels:
                continue
            records.append(
                Record(
                    name=path_text or f"nuimages_{len(records):07d}.jpg",
                    labels=labels,
                    extra={"bytes": image.get("bytes"), "width": width, "height": height, "camera": camera},
                )
            )
    return records


def write_dataset(records: list[Record], output: Path, mode: str, val_fraction: float, seed: int) -> dict:
    import collections as _collections

    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_fraction))
    splits = {"val": shuffled[:val_count], "train": shuffled[val_count:]}
    histogram: dict[str, dict[str, int]] = {}
    for split, items in splits.items():
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        per_class: _collections.Counter[str] = _collections.Counter()
        manifest_lines = []
        for record in items:
            stem = Path(record.name).stem or "image"
            image_path = output / "images" / split / f"{stem}.jpg"
            image_path.write_bytes(record.extra["bytes"])
            lines = [
                f"{index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                for index, cx, cy, w, h in record.labels
            ]
            (output / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            for index, *_ in record.labels:
                per_class[COCO_NAMES[index] if mode == "coco80" else TARGET_CLASSES[index]] += 1
            manifest_lines.append(
                json.dumps(
                    {
                        "split": split,
                        "image": str(image_path.relative_to(output)).replace("\\", "/"),
                        "source": record.name,
                        "camera": record.extra.get("camera"),
                        "instances": len(record.labels),
                    },
                    ensure_ascii=False,
                )
            )
        (output / f"manifest_{split}.jsonl").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        histogram[split] = dict(sorted(per_class.items(), key=lambda item: -item[1]))
    names = COCO_NAMES if mode == "coco80" else TARGET_CLASSES
    (output / "dataset.yaml").write_text(
        "path: {}\ntrain: images/train\nval: images/val\nnames:\n{}\n".format(
            output.as_posix(),
            "\n".join(f"  {index}: {name}" for index, name in enumerate(names)),
        ),
        encoding="utf-8",
    )
    summary = {
        "mode": mode,
        "classes": len(names),
        "train_images": len(splits["train"]),
        "val_images": len(splits["val"]),
        "instances": {split: sum(len(record.labels) for record in items) for split, items in splits.items()},
        "class_histogram": histogram,
    }
    (output / "class_histogram.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--max-shards", type=int, default=12)
    parser.add_argument(
        "--skip-shards",
        default="",
        help="comma separated shard indices to skip, e.g. 0,1,2,3,4 for the known CAM_BACK shards",
    )
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--mode", choices=("coco80", "compact7"), default="coco80")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--front-only", action="store_true", help="keep CAM_FRONT keyframes only")
    parser.add_argument("--min-bicycle", type=int, default=700)
    parser.add_argument("--min-bus", type=int, default=600)
    parser.add_argument("--min-motorcycle", type=int, default=700)
    parser.add_argument("--min-truck", type=int, default=1200)
    parser.add_argument("--common-share", type=float, default=0.35, help="max share of car/person-only frames")
    parser.add_argument("--holdout-names", type=Path, help="frozen holdout filenames; aborts on any collision")
    parser.add_argument("--shard-cache", type=Path, help="keep downloaded shards here instead of a temp dir")
    args = parser.parse_args()

    holdout = set()
    if args.holdout_names:
        holdout = {line.strip() for line in args.holdout_names.read_text(encoding="utf-8").splitlines() if line.strip()}

    rarity_order = [
        class_index("bicycle", args.mode),
        class_index("bus", args.mode),
        class_index("motorcycle", args.mode),
        class_index("truck", args.mode),
        class_index("person", args.mode),
        class_index("car", args.mode),
    ]
    min_instances = {
        class_index("bicycle", args.mode): args.min_bicycle,
        class_index("bus", args.mode): args.min_bus,
        class_index("motorcycle", args.mode): args.min_motorcycle,
        class_index("truck", args.mode): args.min_truck,
    }

    pool: list[Record] = []
    cache = args.shard_cache
    scratch: tempfile.TemporaryDirectory | None = None
    if cache is None:
        scratch = tempfile.TemporaryDirectory()
        cache = Path(scratch.name)
    cache.mkdir(parents=True, exist_ok=True)

    skip = {int(part) for part in args.skip_shards.split(",") if part.strip()}
    for shard in range(args.max_shards):
        if shard in skip:
            print(f"shard {shard}: skipped by --skip-shards", flush=True)
            continue
        target = cache / f"train-{shard:05d}.parquet"
        try:
            if not target.exists():
                download_shard(shard, target)
            records = read_shard(target, args.mode, want_multi_view=not args.front_only)
        except Exception as exc:
            print(f"shard {shard} skipped: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            continue
        pool.extend(records)
        print(f"shard {shard}: +{len(records)} usable frames (pool={len(pool)})", flush=True)
        if len(pool) >= args.limit * 3:
            break

    if args.holdout_names:
        collisions = [record.name for record in pool if Path(record.name).name in holdout]
        if collisions:
            raise SystemExit(f"holdout collision detected: {collisions[:5]}")

    if len(pool) < args.limit:
        print(f"warning: pool has {len(pool)} frames, less than the requested {args.limit}", flush=True)
    chosen = select_balanced(pool, min(args.limit, len(pool)), args.seed, min_instances, rarity_order, args.common_share)

    if args.output.exists():
        shutil.rmtree(args.output)
    summary = write_dataset(chosen, args.output, args.mode, args.val_fraction, args.seed)
    summary["source"] = "finedet/nuimages train split (frozen; no BDD images used)"
    summary["shards_read"] = args.max_shards
    (args.output / "class_histogram.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "class_histogram"}, indent=2))
    if scratch is not None:
        scratch.cleanup()


if __name__ == "__main__":
    main()
