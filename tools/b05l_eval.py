"""B-05L unified BDD50 evaluation for YOLO11n/YOLOv8n/YOLOv8n-P2.

All models see the same top-60% ROI, 512x192 RGB/NCHW/1/255 input.  Detection
matching is strictly one-to-one per image at IoU=0.5.  AP50 is global by
confidence with candidate floor 0.001.  Size groups use original 1280x720
pixel widths (canvas width * 2.5), and their TP total is asserted to equal the
overall TP total.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_ai.traffic_light_tracking import classify_lamp_color  # noqa: E402
from tools.b05k_common import decode_yolo11, prepare_input, size_group  # noqa: E402
from tools.b05k_runtime import ModelRuntime  # noqa: E402

DEFAULT_IMAGES = Path(r"D:\Datasets\adas_edge_training\b03c_rknn_validation_20261021\board_holdout50")
DEFAULT_GT = Path(r"D:\Datasets\adas_edge_training\b03d_gt_hf_20261021")
COLORS = ("RED", "GREEN", "YELLOW", "UNKNOWN")


def load_gt(images_dir: Path, gt_dir: Path, roi_top: float) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, int]]:
    samples = []
    ground_truth: dict[str, list[dict[str, Any]]] = {}
    counts = Counter()
    for gt_path in sorted(gt_dir.glob("*.json")):
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        name = str(data["name"])
        image_path = images_dir / f"{name}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        roi_height = int(round(height * roi_top))
        targets = []
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                if obj.get("category") != "traffic light":
                    continue
                box = obj.get("box2d", {})
                x1 = max(0.0, min(float(width), float(box.get("x1", 0))))
                y1 = max(0.0, min(float(height), float(box.get("y1", 0))))
                x2 = max(0.0, min(float(width), float(box.get("x2", 0))))
                y2 = max(0.0, min(float(height), float(box.get("y2", 0))))
                if x2 <= x1 or y2 <= y1:
                    continue
                counts["all_gt"] += 1
                color = str(obj.get("attributes", {}).get("trafficLightColor", "none")).lower().upper()
                if y2 <= roi_height:
                    counts["roi_gt"] += 1
                    targets.append({
                        "bbox": [x1, y1, x2, y2], "width_px": x2 - x1, "color_gt": color,
                        "occluded": bool(obj.get("attributes", {}).get("occluded", False)),
                        "truncated": bool(obj.get("attributes", {}).get("truncated", False)),
                    })
                elif y1 >= roi_height:
                    counts["outside_roi_gt"] += 1
                else:
                    counts["crossing_roi_gt"] += 1
        ground_truth[name] = targets
        samples.append({"name": name, "path": image_path, "width": width, "height": height})
    return samples, ground_truth, dict(counts)


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_image(gt: list[dict[str, Any]], predictions: list[dict[str, Any]], *, conf: float, iou_threshold: float) -> tuple[list[tuple[dict[str, Any], int]], int, int, int]:
    ordered = sorted((p for p in predictions if float(p["confidence"]) >= conf), key=lambda p: -float(p["confidence"]))
    used: set[int] = set()
    pairs: list[tuple[dict[str, Any], int]] = []
    fp = 0
    for prediction in ordered:
        best_index = None
        best_overlap = 0.0
        for index, target in enumerate(gt):
            if index in used:
                continue
            overlap = iou(prediction["bbox"], target["bbox"])
            if overlap > best_overlap:
                best_overlap, best_index = overlap, index
        if best_index is not None and best_overlap >= iou_threshold:
            used.add(best_index)
            pairs.append((prediction, best_index))
        else:
            fp += 1
    return pairs, len(pairs), fp, len(gt) - len(pairs)


def ranked_matching(gt: dict[str, list[dict[str, Any]]], predictions: dict[str, list[dict[str, Any]]], *, floor: float, iou_threshold: float) -> list[dict[str, Any]]:
    rows = []
    for image_key, items in predictions.items():
        for order, prediction in enumerate(items):
            confidence = float(prediction["confidence"])
            if confidence >= floor:
                rows.append({"confidence": confidence, "image": image_key, "order": order, "bbox": prediction["bbox"]})
    rows.sort(key=lambda row: (-row["confidence"], row["image"], row["order"]))
    used: dict[str, set[int]] = {}
    for row in rows:
        targets = gt.get(row["image"], [])
        image_used = used.setdefault(row["image"], set())
        best_index = None
        best_overlap = 0.0
        for index, target in enumerate(targets):
            if index in image_used:
                continue
            overlap = iou(row["bbox"], target["bbox"])
            if overlap > best_overlap:
                best_overlap, best_index = overlap, index
        if best_index is not None and best_overlap >= iou_threshold:
            image_used.add(best_index)
            row["tp"] = 1
            row["gt_index"] = best_index
            row["gt_bbox"] = targets[best_index]["bbox"]
        else:
            row["tp"] = 0
            row["gt_index"] = None
            row["gt_bbox"] = None
    return rows


def average_precision(rows: list[dict[str, Any]], total_gt: int) -> tuple[float, dict[str, list[float]]]:
    if total_gt == 0 or not rows:
        return 0.0, {"recall": [], "precision": [], "confidence": []}
    tp = np.cumsum([row["tp"] for row in rows], dtype=np.float64)
    fp = np.cumsum([1 - row["tp"] for row in rows], dtype=np.float64)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1.0)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changes = np.flatnonzero(mrec[1:] != mrec[:-1])
    ap = float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))
    grid = np.linspace(0.0, 1.0, 101)
    interp = np.interp(grid, recall, precision, left=precision[0] if len(precision) else 0.0, right=1.0)
    return ap, {
        "recall": grid.tolist(),
        "precision": interp.tolist(),
        "confidence": [float(row["confidence"]) for row in rows],
    }


def confidence_curves(rows: list[dict[str, Any]], total_gt: int, images: int) -> dict[str, Any]:
    thresholds = [0.01 * index for index in range(1, 100)]
    conf = np.asarray([row["confidence"] for row in rows], dtype=np.float64)
    tp = np.asarray([row["tp"] for row in rows], dtype=np.int64)
    order = np.argsort(-conf, kind="stable")
    conf = conf[order]
    tp = tp[order]
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1 - tp)
    total_at = np.searchsorted(-conf, -np.asarray(thresholds), side="right")
    rows_out = []
    for threshold, index in zip(thresholds, total_at):
        n = int(index)
        rows_out.append({
            "conf": threshold,
            "tp": int(cum_tp[n - 1]) if n else 0,
            "fp": int(cum_fp[n - 1]) if n else 0,
            "recall": float(cum_tp[n - 1] / total_gt) if n and total_gt else 0.0,
            "fp_per_image": float(cum_fp[n - 1] / images) if n and images else 0.0,
        })
    return {"thresholds": rows_out}


def fp_budget(rows: list[dict[str, Any]], total_gt: int, images: int, budgets: list[float]) -> list[dict[str, Any]]:
    if not rows:
        return [{"budget_fp_per_image": budget, "threshold": None, "tp": 0, "fp": 0,
                 "recall": 0.0, "precision": 0.0, "fp_per_image": 0.0} for budget in budgets]
    ordered = sorted(rows, key=lambda row: -row["confidence"])
    output = []
    for budget in budgets:
        best = None
        tp = fp = 0
        for row in ordered:
            if row["tp"]:
                tp += 1
            else:
                fp += 1
            if fp / images <= budget:
                best = (row["confidence"], tp, fp)
            else:
                break
        if best is None:
            output.append({"budget_fp_per_image": budget, "threshold": None, "tp": 0, "fp": 0,
                           "recall": 0.0, "precision": 0.0, "fp_per_image": 0.0})
        else:
            threshold, tp, fp = best
            output.append({
                "budget_fp_per_image": budget, "threshold": float(threshold), "tp": tp, "fp": fp,
                "recall": tp / total_gt if total_gt else 0.0,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "fp_per_image": fp / images,
            })
    return output


def color_confusion(frame: np.ndarray, pairs: list[tuple[dict[str, Any], int]], gt: list[dict[str, Any]]) -> dict[str, Any]:
    table = {color: Counter() for color in ("RED", "GREEN", "YELLOW", "none")}
    valid = Counter()
    for prediction, gt_index in pairs:
        target = gt[gt_index]
        gt_color = str(target.get("color_gt", "none")).upper()
        if gt_color not in table:
            gt_color = "none"
        observation = classify_lamp_color(frame, prediction["bbox"])
        predicted = observation.color.value if observation.valid else "UNKNOWN"
        table[gt_color][predicted] += 1
        if observation.valid:
            valid[gt_color] += 1
    return {
        "confusion": {gt_color: dict(counts) for gt_color, counts in table.items()},
        "matched_by_gt_color": {gt_color: sum(counts.values()) for gt_color, counts in table.items()},
        "valid_color_by_gt_color": dict(valid),
        "note": "Only IoU>=0.5 matched predictions at conf=0.25; GT color 'none' is unlit/unknown and excluded from accuracy claims.",
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    samples, gt, gt_counts = load_gt(args.images, args.gt_dir, args.roi_top)
    runtime = ModelRuntime(
        args.backend, args.model, device=args.device,
        precision=args.precision,
        calib_dir=args.calib_dir,
        quantized_algorithm=args.quantized_algorithm,
        score_domain=args.score_domain,
        hybrid_model_input=args.hybrid_model_input,
        hybrid_data_input=args.hybrid_data_input,
        hybrid_cfg=args.hybrid_cfg,
    )
    runtime.validate_canvas(args.input_width, args.input_height)
    warmup_image = cv2.imread(str(samples[0]["path"]))
    warmup_tensor, _ = prepare_input(warmup_image, input_width=args.input_width, input_height=args.input_height,
                                     roi_top=args.roi_top, preprocess=args.preprocess)
    runtime.warmup(warmup_tensor, runs=args.warmup)

    predictions: dict[str, list[dict[str, Any]]] = {}
    inference_times = []
    preprocess_times = []
    input_hashes = {}
    for sample in samples:
        image = cv2.imread(str(sample["path"]))
        t0 = time.perf_counter()
        tensor, transform = prepare_input(image, input_width=args.input_width, input_height=args.input_height,
                                          roi_top=args.roi_top, preprocess=args.preprocess)
        preprocess_times.append((time.perf_counter() - t0) * 1000.0)
        input_hashes[sample["name"]] = hashlib.sha256(tensor.tobytes()).hexdigest()
        raw, infer_ms = runtime.run(tensor)
        inference_times.append(infer_ms)
        decoded = decode_yolo11(raw, conf=args.ap_floor, iou=args.nms_iou, max_det=args.max_det)
        for detection in decoded:
            detection["bbox"] = transform.network_to_frame(detection["bbox"])
        predictions[sample["name"]] = decoded

    runtime.close()

    rows = ranked_matching(gt, predictions, floor=args.ap_floor, iou_threshold=args.match_iou)
    ap, pr = average_precision(rows, gt_counts["roi_gt"])
    groups = {name: {"gt": 0, "tp": 0, "fn": 0} for name in ("<8", "8-16", "16-32", ">32")}
    tp = fp = fn = 0
    color_tables = []
    for sample in samples:
        image = cv2.imread(str(sample["path"]))
        pairs, image_tp, image_fp, image_fn = match_image(
            gt[sample["name"]], predictions[sample["name"]], conf=args.conf, iou_threshold=args.match_iou
        )
        tp += image_tp
        fp += image_fp
        fn += image_fn
        matched = {index for _, index in pairs}
        for index, target in enumerate(gt[sample["name"]]):
            group = size_group(float(target["width_px"]))
            groups[group]["gt"] += 1
            if index in matched:
                groups[group]["tp"] += 1
            else:
                groups[group]["fn"] += 1
        color_tables.append(color_confusion(image, pairs, gt[sample["name"]]))

    group_tp = sum(value["tp"] for value in groups.values())
    if group_tp != tp:
        raise AssertionError(f"size-group TP sum {group_tp} != overall TP {tp}")
    total_roi = gt_counts["roi_gt"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_roi if total_roi else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    full_gt = gt_counts["all_gt"]
    result = {
        "dataset": {
            **gt_counts,
            "images": len(samples),
            "roi_coverage": gt_counts["roi_gt"] / gt_counts["all_gt"] if gt_counts["all_gt"] else 0.0,
            "roi_top_fraction": args.roi_top,
            "coordinate_space": "original BDD image pixels",
            "images_dir": str(args.images.resolve()),
            "gt_dir": str(args.gt_dir.resolve()),
        },
        "model": {
            "path": str(args.model.resolve()),
            "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "backend": args.backend,
            "device": runtime.device,
            "providers": runtime.providers,
            "precision": args.precision if args.backend == "rknn_sim" else None,
            "quantized_algorithm": args.quantized_algorithm if args.backend == "rknn_sim" else None,
            "score_domain": args.score_domain if args.backend == "rknn_sim" else None,
            "hybrid_cfg": str(args.hybrid_cfg.resolve()) if args.hybrid_cfg else None,
            "parameters": sum(value.numel() for value in runtime.model.state_dict().values()) if args.backend == "pytorch" else None,
            "size_bytes": args.model.stat().st_size,
        },
        "input": {"width": args.input_width, "height": args.input_height, "rgb": True, "layout": "NCHW",
                  "dtype": "float32", "scale": "1/255", "preprocess": args.preprocess,
                  "rknn_runtime_layout": "NHWC" if args.backend == "rknn_sim" else None,
                  "rknn_runtime_dtype": "uint8" if args.backend == "rknn_sim" else None},
        "thresholds": {"ap_floor": args.ap_floor, "operating_conf": args.conf, "nms_iou": args.nms_iou, "match_iou": args.match_iou},
        "metrics_roi": {
            "ap50": ap,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "size_groups": groups,
            "tp_sum_size_groups": group_tp,
        },
        "metrics_full_frame": {
            "all_gt": full_gt, "tp": tp, "fn": full_gt - tp,
            "recall": tp / full_gt if full_gt else 0.0,
            "note": "ROI-excluded GT can never be detected; recall is a coverage-adjusted view, not a model failure.",
        },
        "curves": {
            "pr50": pr,
            "confidence": confidence_curves(rows, total_roi, len(samples)),
            "fp_budget": fp_budget(rows, total_roi, len(samples), args.fp_budgets),
        },
        "runtime": {
            "mean_preprocess_ms": statistics.fmean(preprocess_times),
            "mean_inference_ms": statistics.fmean(inference_times),
            "p50_inference_ms": float(np.percentile(inference_times, 50)),
            "p95_inference_ms": float(np.percentile(inference_times, 95)),
        },
        "color": {
            "by_image": color_tables,
            "aggregate": aggregate_colors(color_tables),
        },
        "input_sha256_by_image": input_hashes,
        "predictions_by_image": predictions,
    }
    return result


def aggregate_colors(tables: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Counter] = {}
    matched: Counter = Counter()
    valid: Counter = Counter()
    for table in tables:
        for gt_color, counts in table["confusion"].items():
            agg.setdefault(gt_color, Counter()).update(counts)
        matched.update(table["matched_by_gt_color"])
        valid.update(table["valid_color_by_gt_color"])
    confusion = {gt_color: dict(counts) for gt_color, counts in agg.items()}
    correct = {}
    for gt_color, counts in confusion.items():
        if gt_color == "none":
            continue
        correct[gt_color] = counts.get(gt_color.upper(), 0)
    return {
        "confusion": confusion,
        "matched_by_gt_color": dict(matched),
        "valid_color_by_gt_color": dict(valid),
        "correct_by_gt_color": correct,
        "accuracy_excluding_none": (
            sum(correct.values()) / sum(matched[color] for color in ("RED", "GREEN", "YELLOW"))
            if sum(matched[color] for color in ("RED", "GREEN", "YELLOW")) else 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=("pytorch", "onnx", "rknn_sim"), required=True)
    parser.add_argument("--precision", choices=("fp16", "int8"), default="fp16")
    parser.add_argument("--calib-dir", type=Path)
    parser.add_argument("--quantized-algorithm", choices=("normal", "mmse", "kl_divergence"), default="normal")
    parser.add_argument("--score-domain", choices=("logits", "probabilities"), default="logits")
    parser.add_argument("--hybrid-model-input", type=Path)
    parser.add_argument("--hybrid-data-input", type=Path)
    parser.add_argument("--hybrid-cfg", type=Path)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preprocess", choices=("letterbox", "resize"), default="letterbox")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--ap-floor", type=float, default=0.001)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--roi-top", type=float, default=0.60)
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--input-height", type=int, default=192)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--fp-budgets", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.ap_floor <= args.conf:
        parser.error("thresholds must satisfy 0 <= --ap-floor <= --conf <= 1")
    hybrid_artifacts = (args.hybrid_model_input, args.hybrid_data_input, args.hybrid_cfg)
    if any(hybrid_artifacts) and not all(hybrid_artifacts):
        parser.error("provide all three hybrid artifacts: --hybrid-model-input, --hybrid-data-input, and --hybrid-cfg")
    if args.backend == "rknn_sim" and args.precision == "int8" and args.calib_dir is None and not all(hybrid_artifacts):
        parser.error("--calib-dir is required for RKNN INT8 simulation")
    if all(hybrid_artifacts) and (args.backend != "rknn_sim" or args.precision != "int8"):
        parser.error("hybrid artifacts require --backend rknn_sim --precision int8")
    return args


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    printable = {key: result[key] for key in ("dataset", "model", "thresholds", "metrics_roi", "metrics_full_frame", "runtime", "color")}
    printable["color"] = result["color"]["aggregate"]
    printable["output"] = str(args.output.resolve())
    print(json.dumps(printable, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
