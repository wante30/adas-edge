"""Unified BDD50 evaluation for PyTorch and fixed-shape ONNX traffic-light models."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.b05k_common import decode_yolo11, evaluate_predictions, prepare_input  # noqa: E402
from tools.b05k_runtime import ModelRuntime, resolve_model_path  # noqa: E402


DEFAULT_IMAGES = Path(r"D:\Datasets\adas_edge_training\b03c_rknn_validation_20261021\board_holdout50")
DEFAULT_GT = Path(r"D:\Datasets\adas_edge_training\b03d_gt_hf_20261021")


def load_bdd50(images_dir: Path, gt_dir: Path, roi_top: float) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, int]]:
    samples = []
    ground_truth: dict[str, list[dict[str, Any]]] = {}
    counts = {"all_gt": 0, "roi_gt": 0, "outside_roi_gt": 0, "crossing_roi_gt": 0}
    gt_files = sorted(gt_dir.glob("*.json"))
    if not gt_files:
        raise FileNotFoundError(f"no human-GT JSON files found under {gt_dir}")
    for gt_path in gt_files:
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        name = str(data["name"])
        image_path = images_dir / f"{name}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"cannot read BDD50 source image: {image_path}")
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
                if y2 <= roi_height:
                    counts["roi_gt"] += 1
                    targets.append({"bbox": [x1, y1, x2, y2], "width_px": x2 - x1})
                elif y1 >= roi_height:
                    counts["outside_roi_gt"] += 1
                else:
                    counts["crossing_roi_gt"] += 1
        ground_truth[name] = targets
        samples.append({"name": name, "path": image_path, "width": width, "height": height})
    return samples, ground_truth, counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_variant(
    samples: list[dict[str, Any]],
    all_gt: dict[str, list[dict[str, Any]]],
    runtime: ModelRuntime,
    *,
    preprocess: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    inference_times = []
    preprocess_times = []
    input_hashes = {}
    for index, sample in enumerate(samples):
        image = cv2.imread(str(sample["path"]))
        if image is None:
            raise FileNotFoundError(sample["path"])
        started = time.perf_counter()
        tensor, transform = prepare_input(
            image, input_width=args.input_width, input_height=args.input_height,
            roi_top=args.roi_top, preprocess=preprocess,
        )
        preprocess_times.append((time.perf_counter() - started) * 1000.0)
        input_hashes[sample["name"]] = hashlib.sha256(tensor.tobytes()).hexdigest()
        raw, infer_ms = runtime.run(tensor)
        inference_times.append(infer_ms)
        decoded = decode_yolo11(raw, conf=args.ap_floor, iou=args.nms_iou, max_det=args.max_det)
        for item in decoded:
            item["bbox"] = transform.network_to_frame(item["bbox"])
        predictions[sample["name"]] = decoded

    metrics = evaluate_predictions(
        all_gt, predictions, conf=args.conf, iou=args.match_iou, ap_floor=args.ap_floor
    )
    output = {
        "preprocess": preprocess,
        "metrics": metrics,
        "runtime": {
            "images": len(samples),
            "mean_preprocess_ms": statistics.fmean(preprocess_times) if preprocess_times else None,
            "mean_inference_ms": statistics.fmean(inference_times) if inference_times else None,
            "p95_inference_ms": float(__import__("numpy").percentile(inference_times, 95)) if inference_times else None,
        },
        "input_sha256_by_image": input_hashes,
    }
    if args.save_predictions:
        output["predictions_by_image"] = predictions
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES, help="original BDD50 JPEG directory")
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT, help="human-reviewed per-image GT JSON directory")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=Path)
    model_group.add_argument("--model-manifest", type=Path, help="resolve the original model by SHA-256")
    parser.add_argument("--backend", choices=("pytorch", "onnx"), required=True)
    parser.add_argument("--preprocess", choices=("letterbox", "resize", "ab"), default="letterbox")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--ap-floor", type=float, default=0.001)
    parser.add_argument("--roi-top", type=float, default=0.60)
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--input-height", type=int, default=192)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--save-predictions", action="store_true", help="include per-image prediction records")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.ap_floor <= args.conf:
        parser.error("thresholds must satisfy 0 <= --ap-floor <= --conf <= 1")
    return args


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(
        backend=args.backend, model_path=args.model, manifest_path=args.model_manifest
    )
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {args.output} (use --overwrite)")
    samples, ground_truth, gt_counts = load_bdd50(args.images, args.gt_dir, args.roi_top)
    runtime = ModelRuntime(args.backend, model_path, device=args.device)
    runtime.validate_canvas(args.input_width, args.input_height)
    variants = ("letterbox", "resize") if args.preprocess == "ab" else (args.preprocess,)
    results = {
        "dataset": {
            "images": len(samples),
            **gt_counts,
            "roi_coverage": gt_counts["roi_gt"] / gt_counts["all_gt"] if gt_counts["all_gt"] else 0.0,
            "roi_top_fraction": args.roi_top,
            "coordinate_space": "original BDD image pixels",
            "images_dir": str(args.images.resolve()),
            "human_gt_dir": str(args.gt_dir.resolve()),
        },
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "backend": args.backend,
        "runtime_providers": runtime.providers,
        "device": runtime.device,
        "input": {"width": args.input_width, "height": args.input_height, "rgb": True, "layout": "NCHW", "dtype": "float32", "scale": "1/255"},
        "thresholds": {"ap_floor": args.ap_floor, "operating_conf": args.conf, "nms_iou": args.nms_iou, "match_iou": args.match_iou},
        "variants": {},
    }
    for preprocess in variants:
        sample = cv2.imread(str(samples[0]["path"]))
        warmup_tensor, _ = prepare_input(
            sample, input_width=args.input_width, input_height=args.input_height,
            roi_top=args.roi_top, preprocess=preprocess,
        )
        runtime.warmup(warmup_tensor, runs=args.warmup)
        results["variants"][preprocess] = evaluate_variant(
            samples, ground_truth, runtime, preprocess=preprocess, args=args
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    printable = {
        "output": str(args.output.resolve()),
        "backend": args.backend,
        "device": runtime.device,
        "dataset": results["dataset"],
        "variants": {key: value["metrics"] for key, value in results["variants"].items()},
    }
    print(json.dumps(printable, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
