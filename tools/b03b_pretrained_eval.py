"""Compare BDD-finetuned Ultralytics checkpoints on the B-01R 50-frame sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["ULTRALYTICS_SAFE_LOAD"] = "1"

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.coco_eval_helpers import (  # noqa: E402
    CATEGORY_IDS,
    CLASSES,
    MODEL_MAP,
    build_ground_truth,
    predictions_to_coco,
    read_bdd_rows,
    strip_internal_fields,
    operating_point,
)

BDD_MODEL_ALIASES = {"motor": "motorcycle", "bike": "bicycle"}
COMPARE_CLASSES = ("car", "person", "traffic light")


def canonicalize_model_names(model_names: dict[int, str]) -> dict[int, str]:
    """Translate documented BDD model aliases without merging rider into person."""
    return {int(class_id): BDD_MODEL_ALIASES.get(name, name) for class_id, name in model_names.items()}


def load_frames(bdd_root: Path) -> tuple[list[str], list[tuple[str, object]]]:
    import cv2

    image_root = bdd_root / "images" / "100k" / "val"
    image_names = [path.name for path in sorted(image_root.glob("*.jpg"))[:50]]
    if len(image_names) != 50:
        raise RuntimeError(f"expected 50 sorted BDD validation frames, found {len(image_names)} under {image_root}")
    frames = []
    for name in image_names:
        frame = cv2.imread(str(image_root / name), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"cannot read BDD frame: {image_root / name}")
        frames.append((name, frame))
    return image_names, frames


def collect_predictions(
    weights: Path,
    frames: list[tuple[str, object]],
    *,
    device: str,
    width: int,
    height: int,
    score_floor: float,
    nms_iou: float,
    max_det: int,
) -> tuple[list[tuple[str, list[tuple]]], dict[int, str]]:
    import torch
    from ultralytics import YOLO

    if not weights.is_file():
        raise FileNotFoundError(weights)
    model = YOLO(str(weights), task="detect")
    raw_names = {int(key): str(value) for key, value in model.names.items()}
    rows = []
    args = {
        "imgsz": (height, width),
        "conf": score_floor,
        "iou": nms_iou,
        "max_det": max_det,
        "device": device,
        "half": False,
        "rect": False,
        "verbose": False,
    }
    with torch.inference_mode():
        for name, frame in frames:
            result = model.predict(source=frame, **args)[0]
            boxes = result.boxes
            detections = [
                (int(class_id), float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score))
                for box, class_id, score in zip(
                    boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()
                )
            ]
            rows.append((name, detections))
    del model
    return rows, raw_names


def ap50(detections: list[dict], ground_truth: dict, max_det: int) -> float | None:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return None

    coco_gt = COCO()
    coco_gt.dataset = strip_internal_fields(ground_truth)
    coco_gt.createIndex()
    if detections:
        coco_dt = coco_gt.loadRes(detections)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {"images": coco_gt.dataset["images"], "categories": coco_gt.dataset["categories"], "annotations": []}
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.catIds = [CATEGORY_IDS[name] for name in CLASSES]
    evaluator.params.maxDets = [1, 10, max_det]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats[1])


def summarize_model(
    rows: list[tuple[str, list[tuple]]],
    model_names: dict[int, str],
    image_names: list[str],
    ground_truth: dict,
    *,
    conf: float,
    score_floor: float,
    max_det: int,
) -> dict:
    mapped_names = canonicalize_model_names(model_names)
    detections, detection_stats = predictions_to_coco(rows, image_names, mapped_names, score_floor=score_floor)
    point = operating_point(detections, ground_truth, conf=conf, iou_threshold=0.5)
    total = {
        key: sum(int(point["classes"][name][key]) for name in CLASSES)
        for key in ("tp", "fp", "fn")
    }
    precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
    recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "raw_model_names": {str(key): value for key, value in model_names.items()},
        "evaluation_name_mapping": {str(key): value for key, value in mapped_names.items()},
        "score_floor_detections": detection_stats,
        "operating_point_conf": conf,
        "all_7_classes": {**total, "precision": precision, "recall": recall, "f1": f1},
        "per_class": point["classes"],
        "AP50": ap50(detections, ground_truth, max_det),
        "detections": detections,
    }


def select_visual_frames(ground_truth: dict, image_names: list[str]) -> list[str]:
    counts = {name: {target: 0 for target in COMPARE_CLASSES} for name in image_names}
    id_to_name = {value: key for key, value in CATEGORY_IDS.items()}
    for annotation in ground_truth["annotations"]:
        target = id_to_name[annotation["category_id"]]
        if target in COMPARE_CLASSES:
            counts[ground_truth["images"][annotation["image_id"] - 1]["file_name"]][target] += 1
    ranked = sorted(
        image_names,
        key=lambda name: (
            sum(counts[name][target] > 0 for target in COMPARE_CLASSES),
            min(counts[name]["car"], 5) + min(counts[name]["person"], 3) + min(counts[name]["traffic light"], 5),
            name,
        ),
        reverse=True,
    )
    return ranked[:3]


def write_comparison_image(
    path: Path,
    bdd_root: Path,
    frame_names: list[str],
    rows_by_model: dict[str, list[tuple[str, list[tuple]]]],
    names_by_model: dict[str, dict[int, str]],
    ground_truth: dict,
    *,
    conf: float,
) -> None:
    import cv2
    import numpy as np

    width, height = 640, 360
    headings = ("BDD ground truth", "COCO YOLO11n", "BDD YOLO11n", "BDD YOLOv8n")
    colors = {
        "car": (40, 190, 255),
        "person": (60, 220, 60),
        "traffic light": (255, 150, 40),
        "truck": (200, 80, 240),
        "bus": (40, 220, 220),
        "motorcycle": (230, 230, 50),
        "bicycle": (230, 120, 120),
    }
    canvas = np.zeros((len(frame_names) * (height + 30) + 56, len(headings) * width, 3), dtype=np.uint8)
    for column, title in enumerate(headings):
        cv2.putText(canvas, title, (column * width + 8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 235, 235), 1, cv2.LINE_AA)
    image_to_id = {entry["file_name"]: entry["id"] for entry in ground_truth["images"]}
    gt_by_image: dict[int, list[dict]] = {}
    for annotation in ground_truth["annotations"]:
        gt_by_image.setdefault(annotation["image_id"], []).append(annotation)
    prediction_index = {
        label: {name: values for name, values in rows}
        for label, rows in rows_by_model.items()
    }
    for row_index, image_name in enumerate(frame_names):
        source = cv2.imread(str(bdd_root / "images" / "100k" / "val" / image_name), cv2.IMREAD_COLOR)
        if source is None:
            continue
        source = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
        top = 56 + row_index * (height + 30)
        cv2.putText(canvas, image_name, (8, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        gt_tile = source.copy()
        image_id = image_to_id[image_name]
        for annotation in gt_by_image.get(image_id, []):
            target = next(name for name, category_id in CATEGORY_IDS.items() if category_id == annotation["category_id"])
            x, y, w, h = annotation["bbox"]
            box = tuple(int(round(v * 0.5)) for v in (x, y, x + w, y + h))
            color = colors.get(target, (220, 220, 220))
            cv2.rectangle(gt_tile, box[:2], box[2:], color, 1)
        canvas[top : top + height, :width] = gt_tile
        for column, label in enumerate(("coco_yolo11n", "bdd_yolo11n", "bdd_yolov8n"), start=1):
            tile = source.copy()
            names = canonicalize_model_names(names_by_model[label])
            for model_class, x1, y1, x2, y2, score in prediction_index[label].get(image_name, []):
                if score < conf:
                    continue
                target = MODEL_MAP.get(names.get(int(model_class), ""))
                if target is None:
                    continue
                box = tuple(int(round(v * 0.5)) for v in (x1, y1, x2, y2))
                color = colors.get(target, (220, 220, 220))
                cv2.rectangle(tile, box[:2], box[2:], color, 1)
                cv2.putText(tile, f"{target} {score:.2f}", (box[0], max(12, box[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.31, color, 1, cv2.LINE_AA)
            left = column * width
            canvas[top : top + height, left : left + width] = tile
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write comparison image: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdd-root", type=Path, required=True)
    parser.add_argument("--coco-yolo11", type=Path, required=True)
    parser.add_argument("--bdd-yolo11", type=Path, required=True)
    parser.add_argument("--bdd-yolov8", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.width != 512 or args.height != 288:
        parser.error("B-03B requires the fixed 512x288 canvas")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be nonnegative and iterations must be positive")
    args.outdir.mkdir(parents=True, exist_ok=False)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; B-03B requires RTX 4060 local inference")

    image_names, frames = load_frames(args.bdd_root)
    rows = read_bdd_rows(args.bdd_root / "bdd100k.csv")
    ground_truth, ground_truth_stats = build_ground_truth(rows, image_names, width=1280, height=720)
    model_paths = {
        "coco_yolo11n": args.coco_yolo11,
        "bdd_yolo11n": args.bdd_yolo11,
        "bdd_yolov8n": args.bdd_yolov8,
    }
    result = {
        "protocol": {
            "sample": "first 50 .jpg filenames in lexical order, matching B-01R-Lite",
            "images": len(image_names),
            "input": [1, 3, args.height, args.width],
            "device": torch.cuda.get_device_name(0),
            "backend": "Ultralytics PyTorch CUDA FP32",
            "confidence": args.conf,
            "prediction_score_floor": args.score_floor,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
            "match_iou": 0.5,
            "ground_truth": ground_truth_stats,
            "rider_mapping": "kept separate and excluded from the person target",
        },
        "models": {},
        "benchmarks": {},
    }
    rows_by_model = {}
    names_by_model = {}
    for label, weights in model_paths.items():
        print(f"predicting {label}: {weights}", flush=True)
        predictions, names = collect_predictions(
            weights,
            frames,
            device=args.device,
            width=args.width,
            height=args.height,
            score_floor=args.score_floor,
            nms_iou=args.nms_iou,
            max_det=args.max_det,
        )
        rows_by_model[label] = predictions
        names_by_model[label] = names
        metrics = summarize_model(
            predictions,
            names,
            image_names,
            ground_truth,
            conf=args.conf,
            score_floor=args.score_floor,
            max_det=args.max_det,
        )
        result["models"][label] = {key: value for key, value in metrics.items() if key != "detections"}
        (args.outdir / f"{label}_predictions.json").write_text(
            json.dumps(
                {
                    "model": str(weights),
                    "model_names": {str(key): value for key, value in names.items()},
                    "per_image": [
                        {"image": name, "detections": [list(detection) for detection in values]}
                        for name, values in predictions
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        result["models"][label]["detections"] = metrics["detections"]

    from tools.benchmark_b02_models import benchmark_ultralytics_cuda
    for label, weights in model_paths.items():
        print(f"benchmarking {label}", flush=True)
        result["benchmarks"][label] = benchmark_ultralytics_cuda(
            weights, frames, warmup=args.warmup, iterations=args.iterations
        )

    visual_frames = select_visual_frames(ground_truth, image_names)
    write_comparison_image(
        args.outdir / "comparison_grid.jpg",
        args.bdd_root,
        visual_frames,
        rows_by_model,
        names_by_model,
        ground_truth,
        conf=args.conf,
    )
    result["visual_frames"] = visual_frames
    (args.outdir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"outdir": str(args.outdir), "visual_frames": visual_frames, "models": result["models"], "benchmarks": result["benchmarks"]}, indent=2))


if __name__ == "__main__":
    main()
