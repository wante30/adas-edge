"""Evaluate a YOLO checkpoint on the frozen BDD100K image holdout.

The 1000-image holdout is never used for training. Matching keeps the historical
protocol - one-to-one per class, confidence >= 0.25, IoU >= 0.5 - so new numbers
stay comparable with the existing baseline reports, and adds the diagnostics the
road-domain pilot needs: macro recall, per-class F1, small-object recall buckets
and wall-clock throughput.

Two things this evaluator refuses to leave implicit:

``--imgsz``      accepts a square size (``320``) or a rectangular one (``512x288``).
``--geometry``   ``fixed`` letterboxes to exactly that canvas; ``minrect`` lets
                 Ultralytics shrink the canvas to the minimum stride-aligned
                 rectangle for the batch aspect ratio.

The distinction matters because Ultralytics sets ``rect=True`` by default for
prediction, so a request for ``320`` actually feeds a 16:9 road frame at
``192x320`` - 61k pixels, not 320x320. The report records the shape the network
really received, so a "320 baseline" can never silently mean two different
things. ``fixed`` is the default because that is what a deployed RKNN graph gets:
one frozen input shape, padding included.

Latency is reported as wall-clock milliseconds per image for the whole pass.
``Results.speed`` is not used: it is a per-batch profiler reading that comes out
identical for every image and produced a bogus 3.4x penalty for the rectangular
shape. Per-shape latency comparison lives in ``tools/benchmark_yolo_shapes.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

CLASSES = ["car", "truck", "bus", "person", "motorcycle", "bicycle", "traffic light"]
BDD_MAP = {"car": "car", "truck": "truck", "bus": "bus", "pedestrian": "person", "person": "person", "motorcycle": "motorcycle", "bicycle": "bicycle", "traffic light": "traffic light"}
MODEL_MAP = {"person": "person", "bicycle": "bicycle", "car": "car", "motorcycle": "motorcycle", "bus": "bus", "truck": "truck", "traffic_light": "traffic light", "traffic light": "traffic light"}

# COCO object-size thresholds in original-pixel area.
SMALL_AREA = 32 * 32
MEDIUM_AREA = 96 * 96
BUCKETS = ("small", "medium", "large")


def parse_imgsz(value: str | int) -> int | tuple[int, int]:
    """Return an Ultralytics ``imgsz``: an int, or ``(height, width)``."""
    if isinstance(value, int):
        return value
    text = str(value).strip().lower().replace("×", "x").replace("*", "x")
    if "x" not in text:
        return int(text)
    width, height = (int(part) for part in text.split("x", 1))
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError(f"image size must be >=32 and divisible by 32, got {width}x{height}")
    return (height, width)


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b - inter else 0.0


def bucket_of(box: list[float]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area < SMALL_AREA:
        return "small"
    return "medium" if area < MEDIUM_AREA else "large"


def match_class(preds: list[list[float]], ground_truth: list[list[float]], iou_threshold: float) -> tuple[int, int, set[int]]:
    """Greedy one-to-one matching. Returns (false positives, true positives, matched GT)."""
    pairs = sorted(
        (
            (iou(pred, gt), pred_index, gt_index)
            for pred_index, pred in enumerate(preds)
            for gt_index, gt in enumerate(ground_truth)
        ),
        reverse=True,
    )
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    for score, pred_index, gt_index in pairs:
        if score < iou_threshold or pred_index in matched_pred or gt_index in matched_gt:
            continue
        matched_pred.add(pred_index)
        matched_gt.add(gt_index)
    return len(preds) - len(matched_pred), len(matched_pred), matched_gt


def evaluate(
    predictions: list[dict[str, list[list[float]]]],
    ground_truth: list[dict[str, dict[str, list[list[float]]]]],
    iou_threshold: float = 0.5,
) -> dict:
    """Aggregate per-class counts, macro recall and GT-recall per size bucket."""
    totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASSES}
    buckets = {(name, bucket): {"gt": 0, "matched": 0} for name in CLASSES for bucket in BUCKETS}
    for image_preds, image_gt in zip(predictions, ground_truth):
        for name in CLASSES:
            preds = image_preds.get(name, [])
            truth = image_gt.get(name, [])
            false_positives, true_positives, matched_gt = match_class(preds, truth, iou_threshold)
            totals[name]["tp"] += true_positives
            totals[name]["fp"] += false_positives
            totals[name]["fn"] += len(truth) - len(matched_gt)
            for index, box in enumerate(truth):
                entry = buckets[(name, bucket_of(box))]
                entry["gt"] += 1
                entry["matched"] += int(index in matched_gt)
    for name, value in totals.items():
        value["precision"] = value["tp"] / (value["tp"] + value["fp"]) if value["tp"] + value["fp"] else 0.0
        value["recall"] = value["tp"] / (value["tp"] + value["fn"]) if value["tp"] + value["fn"] else 0.0
        precision, recall = value["precision"], value["recall"]
        value["f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    bucket_report = {
        bucket: {
            "gt": sum(entry["gt"] for (_, name_bucket), entry in buckets.items() if name_bucket == bucket),
            "matched": sum(entry["matched"] for (_, name_bucket), entry in buckets.items() if name_bucket == bucket),
        }
        for bucket in BUCKETS
    }
    for bucket, value in bucket_report.items():
        value["recall"] = value["matched"] / value["gt"] if value["gt"] else 0.0
    per_class_buckets = {
        f"{name}/{bucket}": (
            buckets[(name, bucket)]["matched"] / buckets[(name, bucket)]["gt"] if buckets[(name, bucket)]["gt"] else None
        )
        for name in CLASSES
        for bucket in BUCKETS
    }
    macro_recall = sum(value["recall"] for value in totals.values()) / len(CLASSES)
    macro_f1 = sum(value["f1"] for value in totals.values()) / len(CLASSES)
    return {
        "classes": totals,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "size_buckets": bucket_report,
        "per_class_size_recall": per_class_buckets,
    }


def format_shape(shape: list[int] | tuple[int, ...]) -> str:
    """Render a recorded network input shape as ``CHW`` or ``WxH``."""
    if len(shape) == 3:
        return f"{shape[2]}x{shape[1]} ({shape[0]}ch)"
    return "x".join(str(value) for value in shape)


def load_ground_truth(csv_path: Path, limit: int, image_root: Path) -> tuple[list[str], list[dict[str, dict[str, list[list[float]]]]]]:
    grouped: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            mapped = BDD_MAP.get(row["class_name"])
            if mapped is None:
                continue
            grouped[Path(row["image_path"]).name][mapped].append(
                [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])]
            )
    images = sorted(image_root.glob("*.jpg"))[:limit]
    return [image.name for image in images], [dict(grouped.get(image.name, {})) for image in images]


def render_table(report: dict) -> str:
    lines = [
        f"# {report['label']} @ {report['size']} ({report['geometry']}) on {report['images']} frozen BDD holdout frames",
        "",
        f"- requested size: {report['size']} | geometry: {report['geometry']}",
        f"- observed network input: {' / '.join(format_shape(shape) for shape in report['observed_input_shapes'])}",
        "",
        "| class | P | R | F1 | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in CLASSES:
        value = report["classes"][name]
        lines.append(
            f"| {name} | {value['precision']:.3f} | {value['recall']:.3f} | {value['f1']:.3f} "
            f"| {value['tp']} | {value['fp']} | {value['fn']} |"
        )
    lines.append(
        f"| **macro** | - | **{report['macro_recall']:.3f}** | **{report['macro_f1']:.3f}** | | | |"
    )
    lines.append("")
    lines.append("| size bucket | GT | matched | recall |")
    lines.append("| --- | ---: | ---: | ---: |")
    for bucket, value in report["size_buckets"].items():
        lines.append(f"| {bucket} | {value['gt']} | {value['matched']} | {value['recall']:.3f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bdd-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, help="optional Markdown table output")
    parser.add_argument("--imgsz", default="320", help="square size or WxH, e.g. 320, 416, 512x288")
    parser.add_argument(
        "--geometry",
        choices=("fixed", "minrect"),
        default="fixed",
        help="fixed = letterbox to exactly --imgsz; minrect = minimum stride-aligned rectangle",
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-match", type=float, default=0.5)
    parser.add_argument("--label", help="report label, defaults to the model stem plus the size")
    args = parser.parse_args()

    try:
        imgsz = parse_imgsz(args.imgsz)
    except ValueError as exc:
        parser.error(str(exc))

    from ultralytics import YOLO

    root = args.bdd_root
    names, ground_truth = load_ground_truth(root / "bdd100k.csv", args.limit, root / "images" / "100k" / "val")
    image_paths = [str(root / "images" / "100k" / "val" / name) for name in names]

    model = YOLO(str(args.model))
    predict_args = {
        "imgsz": imgsz,
        "conf": args.conf,
        "device": args.device,
        "rect": args.geometry == "minrect",
        "verbose": False,
    }
    # Warm the predictor before attaching the recorder, otherwise the warmup
    # tensor would be reported as if it were a measured input shape.
    model.predict(source=image_paths[0], **predict_args)

    observed: set[tuple[int, ...]] = set()

    def record_input(_module, inputs, _kwargs=None):
        tensor = inputs[0] if inputs else None
        if hasattr(tensor, "shape"):
            observed.add(tuple(tensor.shape[1:]))
        return None  # returning anything else would replace the module input

    handle = model.model.register_forward_pre_hook(record_input)
    predictions: list[dict[str, list[list[float]]]] = []
    detections = 0
    started = time.perf_counter()
    try:
        for result in model.predict(source=image_paths, stream=True, **predict_args):
            per_image: dict[str, list[list[float]]] = defaultdict(list)
            for box, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                mapped = MODEL_MAP.get(model.names[int(cls)])
                if mapped:
                    per_image[mapped].append(box)
            detections += sum(len(value) for value in per_image.values())
            predictions.append(dict(per_image))
    finally:
        handle.remove()
    elapsed = time.perf_counter() - started

    metrics = evaluate(predictions, ground_truth, iou_threshold=args.iou_match)
    label = args.label or f"{Path(args.model).stem}-{args.imgsz}"
    report = {
        "dataset": "BDD100K validation-only frozen holdout (1000-image deterministic slice)",
        "label": label,
        "model": str(args.model),
        "images": len(names),
        "size": args.imgsz,
        "geometry": args.geometry,
        "observed_input_shapes": sorted([list(shape) for shape in observed]),
        "conf": args.conf,
        "iou_match": args.iou_match,
        "detections": detections,
        "wall_ms_per_image": elapsed * 1000 / len(names) if names else None,
        "wall_total_s": elapsed,
        **metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    table = render_table(report)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
