"""COCO-style evaluation of a YOLO checkpoint on the frozen BDD100K holdout.

This is the standard-metric evaluator. ``tools/evaluate_bdd_yolo.py`` keeps its
historical one-to-one macro-recall protocol; this tool answers the question a
paper reviewer would ask, using pycocotools:

    mAP@0.5, mAP@0.5:0.95, mAP small/medium/large, AR
    per-class AP50 and AP50:95
    plus, at an explicit operating confidence, per-class precision/recall/F1

The three families of numbers are different quantities and are never merged:

    macro recall  - greedy one-to-one per class at one confidence, project history
    AP50          - area under the precision/recall curve at IoU 0.5
    AP50:95       - the same averaged over ten IoU thresholds, COCO's headline

The input canvas is always explicit: ``--width`` and ``--height`` are separate, so
a rectangular deployment shape can never be confused with a square one. The tool
records the tensor shape the network actually received.

Predictions are dumped at a low score floor so a single inference pass serves both
mAP and any later threshold diagnostic; ``--reuse-predictions`` skips the model
entirely when that dump already exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):  # allow `python tools/evaluate_bdd_coco.py` from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.coco_eval_helpers import (
    AREA_INDEX,
    CATEGORY_IDS,
    CLASSES,
    average_precision,
    average_precision_over_iou,
    build_ground_truth,
    operating_point,
    predictions_to_coco,
    read_bdd_rows,
    strip_internal_fields,
)


def default_image_root(bdd_root: Path) -> Path:
    return bdd_root / "images" / "100k" / "val"


def collect_predictions(
    model_path: Path,
    image_paths: list[str],
    image_names: list[str],
    width: int,
    height: int,
    device: str,
    score_floor: float,
    max_det: int,
) -> tuple[list[tuple[str, list[tuple[int, float, float, float, float, float]]]], dict]:
    """Run the model over the holdout on a static canvas, one image per forward pass.

    Each image is submitted on its own. Ultralytics' ``stream=True`` path batches an
    entire list source into a single forward (a 1000-image list becomes one
    ``[1000, 3, H, W]`` tensor) and ignores ``batch=1``, which would make the
    reported tensor shape a batch shape and the reported latency a batched
    throughput rather than a single-frame cost. Looping one image at a time keeps
    both quantities unambiguous. Accuracy is unaffected either way, because a fixed
    canvas letterboxes every frame to the same HxW.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    predict_args = {
        "imgsz": (height, width),
        "conf": score_floor,
        "max_det": max_det,
        "device": device,
        "rect": False,  # a deployed graph has one frozen canvas, never a dynamic minimum rectangle
        "verbose": False,
    }
    model.predict(source=image_paths[0], **predict_args)  # warm up before recording shapes

    observed: set[tuple[int, ...]] = set()

    def record_input(_module, inputs, _kwargs=None):
        tensor = inputs[0] if inputs else None
        if hasattr(tensor, "shape"):
            observed.add(tuple(int(value) for value in tensor.shape))
        return None  # anything else would replace the module input

    handle = model.model.register_forward_pre_hook(record_input)
    per_image: list[tuple[str, list[tuple[int, float, float, float, float, float]]]] = []
    started = time.perf_counter()
    try:
        for name, path in zip(image_names, image_paths):
            result = model.predict(source=path, **predict_args)[0]
            rows = [
                (int(cls), float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score))
                for box, cls, score in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                )
            ]
            per_image.append((name, rows))
    finally:
        handle.remove()
    elapsed = time.perf_counter() - started
    shapes = sorted(observed)
    per_sample = sorted({tuple([1, *shape[1:]]) for shape in shapes}) if all(shape[0] == 1 for shape in shapes) else []
    runtime = {
        "wall_ms_per_image": elapsed * 1000 / len(image_names) if image_names else None,
        "wall_total_s": elapsed,
        "timing_scope": "PC FP32, one image per forward pass, includes decode and postprocess",
        "observed_input_shapes": [list(shape) for shape in shapes],
        "per_sample_shapes": [list(shape) for shape in per_sample],
        "model_names": {int(index): name for index, name in model.names.items()},
    }
    return per_image, runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bdd-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--width", type=int, required=True, help="static canvas width")
    parser.add_argument("--height", type=int, required=True, help="static canvas height")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25, help="operating confidence for P/R/F1")
    parser.add_argument("--score-floor", type=float, default=0.001, help="detections kept for the mAP sweep")
    parser.add_argument("--max-det", type=int, default=100, help="COCO standard is 100 detections per image")
    parser.add_argument("--label")
    parser.add_argument("--dump-predictions", type=Path)
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()

    if args.width < 32 or args.height < 32 or args.width % 32 or args.height % 32:
        parser.error(f"canvas must be >=32 and divisible by 32, got {args.width}x{args.height}")
    if args.reuse_predictions and not args.dump_predictions:
        parser.error("--reuse-predictions requires --dump-predictions")

    image_root = default_image_root(args.bdd_root)
    image_names = [path.name for path in sorted(image_root.glob("*.jpg"))[: args.limit]]
    image_paths = [str(image_root / name) for name in image_names]
    if not image_names:
        raise SystemExit(f"no holdout images under {image_root}")

    rows = read_bdd_rows(args.bdd_root / "bdd100k.csv")
    coco_gt, gt_stats = build_ground_truth(rows, image_names, width=1280, height=720)

    label = args.label or f"{args.model.stem}-{args.width}x{args.height}"
    runtime: dict = {}
    if args.reuse_predictions and args.dump_predictions.exists():
        payload = json.loads(args.dump_predictions.read_text(encoding="utf-8"))
        model_names = {int(key): value for key, value in payload["model_names"].items()}
        per_image = [
            (entry["image"], [tuple(row) for row in entry["detections"]]) for entry in payload["per_image"]
        ]
        runtime = {"reused_from": str(args.dump_predictions), "observed_input_shapes": payload.get("observed_input_shapes")}
        print(f"reused predictions from {args.dump_predictions}")
    else:
        per_image, runtime = collect_predictions(
            args.model, image_paths, image_names, args.width, args.height,
            args.device, args.score_floor, args.max_det,
        )
        model_names = runtime["model_names"]
        if args.dump_predictions:
            args.dump_predictions.parent.mkdir(parents=True, exist_ok=True)
            args.dump_predictions.write_text(
                json.dumps(
                    {
                        "model": str(args.model),
                        "width": args.width,
                        "height": args.height,
                        "score_floor": args.score_floor,
                        "max_det": args.max_det,
                        "observed_input_shapes": runtime["observed_input_shapes"],
                        "model_names": {str(key): value for key, value in model_names.items()},
                        "per_image": [{"image": name, "detections": [list(row) for row in rows]} for name, rows in per_image],
                    }
                ),
                encoding="utf-8",
            )

    detections, detection_stats = predictions_to_coco(per_image, image_names, model_names, args.score_floor)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco = COCO()
    coco.dataset = strip_internal_fields(coco_gt)
    coco.createIndex()
    if detections:
        coco_dt = coco.loadRes(detections)
    else:
        # pycocotools cannot load an empty result list; an explicitly empty
        # detection set is a legitimate outcome (conf floor above every score).
        coco_dt = COCO()
        coco_dt.dataset = {"images": coco.dataset["images"], "categories": coco.dataset["categories"], "annotations": []}
        coco_dt.createIndex()
    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
    coco_eval.params.catIds = [CATEGORY_IDS[name] for name in CLASSES]
    coco_eval.params.areaRng = [[0.0, 1e10], [0.0, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e10]]
    coco_eval.params.maxDets = [1, 10, args.max_det]
    coco_eval.evaluate()
    coco_eval.accumulate()
    # summarize() prints the standard table and populates .stats (it returns None).
    coco_eval.summarize()
    statistics = list(coco_eval.stats)

    precision = coco_eval.eval["precision"]
    per_class = {}
    for index, name in enumerate(CLASSES):
        per_class[name] = {
            "AP50": average_precision(precision, 0, AREA_INDEX["all"], index),
            "AP50_95": average_precision_over_iou(precision, range(10), AREA_INDEX["all"], index),
            "AP_small": average_precision(precision, 0, AREA_INDEX["small"], index),
            "AP_medium": average_precision(precision, 0, AREA_INDEX["medium"], index),
            "AP_large": average_precision(precision, 0, AREA_INDEX["large"], index),
            "gt_instances": gt_stats["per_class_instances"][name],
            "detections_at_floor": detection_stats["per_class_detections"][name],
        }

    point = operating_point(detections, coco_gt, conf=args.conf)
    for name in CLASSES:
        per_class[name]["precision_at_conf"] = point["classes"][name]["precision"]
        per_class[name]["recall_at_conf"] = point["classes"][name]["recall"]
        per_class[name]["f1_at_conf"] = point["classes"][name]["f1"]

    report = {
        "label": label,
        "model": str(args.model),
        "protocol": {
            "holdout": "BDD100K validation-only, deterministic first-N-by-name slice, never trained on",
            "images": len(image_names),
            "canvas_width": args.width,
            "canvas_height": args.height,
            "canvas_pixels": args.width * args.height,
            "canvas": f"{args.width}x{args.height}",
            "geometry": "fixed",
            "observed_input_shapes": runtime.get("observed_input_shapes"),
            "observed_per_sample_shape": runtime.get("per_sample_shapes"),
            "max_det": args.max_det,
            "score_floor": args.score_floor,
            "operating_conf": args.conf,
            "iou_match_operating_point": 0.5,
            "class_mapping": "BDD100K -> 7 competition classes; rider/traffic sign/train/trailer/other dropped",
            "metric_note": "macro_recall is an operating-point quantity and is NOT mAP50 or mAP50:95",
        },
        "ground_truth": gt_stats,
        "detections": detection_stats,
        "mAP50_95": statistics[0],
        "mAP50": statistics[1],
        "mAP75": statistics[2],
        "mAP_small": statistics[3],
        "mAP_medium": statistics[4],
        "mAP_large": statistics[5],
        "AR_1": statistics[6],
        "AR_10": statistics[7],
        "AR_100": statistics[8],
        "per_class": per_class,
        "operating_point": point,
        "runtime": runtime,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"# {label} on the frozen BDD100K holdout ({len(image_names)} images)",
        "",
        f"- static canvas `{args.width}x{args.height}` ({args.width * args.height} px), "
        f"observed network input `{runtime.get('observed_input_shapes')}`",
        f"- max_det {args.max_det}, score floor {args.score_floor}, operating conf {args.conf}",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| mAP@0.5:0.95 | {report['mAP50_95']:.4f} |",
        f"| mAP@0.5 | {report['mAP50']:.4f} |",
        f"| mAP@0.75 | {report['mAP75']:.4f} |",
        f"| mAP small | {report['mAP_small']:.4f} |",
        f"| mAP medium | {report['mAP_medium']:.4f} |",
        f"| mAP large | {report['mAP_large']:.4f} |",
        f"| macro recall @conf {args.conf} | {point['macro_recall']:.4f} |",
        f"| macro F1 @conf {args.conf} | {point['macro_f1']:.4f} |",
        "",
        "| class | AP50 | AP50:95 | P@conf | R@conf | F1@conf | GT |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in CLASSES:
        entry = per_class[name]
        lines.append(
            f"| {name} | {entry['AP50']:.4f} | {entry['AP50_95']:.4f} | {entry['precision_at_conf']:.3f} "
            f"| {entry['recall_at_conf']:.3f} | {entry['f1_at_conf']:.3f} | {entry['gt_instances']} |"
        )
    lines += ["", "| size bucket | GT | matched | recall |", "| --- | ---: | ---: | ---: |"]
    for bucket in ("small", "medium", "large"):
        value = point["size_buckets"][bucket]
        lines.append(f"| {bucket} | {value['gt']} | {value['matched']} | {value['recall']:.4f} |")
    text = "\n".join(lines)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
