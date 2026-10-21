"""Compare split-output YOLO RKNN models on one shared set of board images."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

COCO_FOCUS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
}
BDD_SOURCE_NAMES = {
    0: "person",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motor",
    7: "bike",
    8: "traffic light",
    9: "traffic sign",
}
BDD_NAME_ALIASES = {"motor": "motorcycle", "bike": "bicycle"}
BDD_NAMES = {class_id: BDD_NAME_ALIASES.get(name, name) for class_id, name in BDD_SOURCE_NAMES.items()}
COMMON_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle", "traffic light"}


def summarize_latency(values: Sequence[float]) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("latency samples must be a non-empty finite sequence")
    return {
        "mean_ms": round(float(samples.mean()), 3),
        "p50_ms": round(float(np.percentile(samples, 50)), 3),
        "p95_ms": round(float(np.percentile(samples, 95)), 3),
        "min_ms": round(float(samples.min()), 3),
        "max_ms": round(float(samples.max()), 3),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--baseline-kind", choices=("coco", "bdd"), default="coco")
    parser.add_argument("--candidate-kind", choices=("coco", "bdd"), default="bdd")
    parser.add_argument("--baseline-label", default="YOLO11n-production")
    parser.add_argument("--candidate-label", default="BDD-YOLOv8n-INT8")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--pipeline-warmup", type=int, default=20)
    parser.add_argument("--pipeline-runs", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.limit < 1 or args.warmup < 0 or args.pipeline_warmup < 0:
        parser.error("--limit must be positive and warmup counts cannot be negative")
    if args.runs < 1 or args.pipeline_runs < 1:
        parser.error("run counts must be positive")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error("--conf and --iou must be in [0, 1]")
    if args.output.exists():
        parser.error(f"refusing to overwrite result: {args.output}")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_npu_status() -> dict[str, str | None]:
    root = Path("/sys/class/devfreq/fde40000.npu")
    result: dict[str, str | None] = {}
    for field in ("load", "cur_freq"):
        try:
            result[field] = (root / field).read_text(encoding="utf-8").strip()
        except OSError:
            result[field] = None
    return result


def _load_samples(args: argparse.Namespace) -> list[dict[str, object]]:
    import cv2

    from tools.profile_rknn_single_stream import letterbox_bgr

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("canvas") != "512x288" or manifest.get("layout") != "NHWC":
        raise ValueError("expected the existing 512x288 NHWC calibration manifest")
    if manifest.get("channel_order") != "RGB" or manifest.get("dtype") != "uint8":
        raise ValueError("expected RGB uint8 input tensors")

    entries = manifest.get("entries", [])[: args.limit]
    if not entries:
        raise ValueError("input manifest contains no samples")

    samples: list[dict[str, object]] = []
    for entry in entries:
        tensor_path = args.inputs_dir / entry["file"]
        tensor = np.load(tensor_path, allow_pickle=False)
        if tensor.shape == (288, 512, 3):
            tensor = tensor[None, ...]
        if tensor.shape != (1, 288, 512, 3) or tensor.dtype != np.uint8:
            raise ValueError(f"{tensor_path}: expected [1,288,512,3] uint8, got {tensor.shape} {tensor.dtype}")

        image_path = args.images_dir / entry["image"]
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"cannot read image: {image_path}")
        canvas, scale, pad_x, pad_y = letterbox_bgr(frame, (512, 288))
        if not np.array_equal(canvas[None, ...], tensor):
            raise ValueError(f"{entry['image']}: deployed letterbox does not match the saved RGB tensor")
        samples.append(
            {
                "name": str(entry["image"]),
                "tensor": np.ascontiguousarray(tensor),
                "frame": frame,
                "scale": float(scale),
                "pad_x": float(pad_x),
                "pad_y": float(pad_y),
            }
        )
    return samples


def _resources(process: object | None) -> dict[str, float | int | None]:
    if process is None:
        return {"process_rss_mb": None, "process_cpu_percent": None, "system_cpu_percent": None}
    try:
        import psutil

        return {
            "process_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
            "process_cpu_percent": round(float(process.cpu_percent(None)), 2),
            "system_cpu_percent": round(float(psutil.cpu_percent(None)), 2),
        }
    except Exception:
        return {"process_rss_mb": None, "process_cpu_percent": None, "system_cpu_percent": None}


def _run_model(
    *,
    label: str,
    model_path: Path,
    classes: dict[int, str],
    model_kind: str,
    samples: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    import cv2
    from rknnlite.api import RKNNLite
    from deployment.rknn.decode_yolo_output import decode_outputs
    from tools.profile_rknn_single_stream import letterbox_bgr

    try:
        import psutil

        process = psutil.Process()
        psutil.cpu_percent(None)
    except ImportError:
        psutil = None
        process = None
    runner = RKNNLite(verbose=False)
    npu_status_before = _read_npu_status()
    resources_before = _resources(process)
    if runner.load_rknn(str(model_path)) != 0:
        raise RuntimeError(f"load_rknn failed: {model_path}")
    if runner.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
        raise RuntimeError(f"init_runtime failed: {model_path}")
    resources_after_init = _resources(process)

    npu_samples: list[float] = []
    pipeline_samples: list[dict[str, float]] = []
    predictions: dict[str, list[dict[str, object]]] = {}
    tensor_shapes: list[list[int]] = []
    tensor_dtypes: list[str] = []
    tensor_finite: list[bool] = []
    resources_after_measurement: dict[str, float | int | None] = {}
    npu_status_after: dict[str, str | None] = {}

    try:
        for index in range(args.warmup):
            sample = samples[index % len(samples)]
            runner.inference(inputs=[sample["tensor"]], data_format="nhwc")
        for index in range(args.runs):
            sample = samples[index % len(samples)]
            started = time.perf_counter()
            outputs = runner.inference(inputs=[sample["tensor"]], data_format="nhwc")
            npu_samples.append((time.perf_counter() - started) * 1000.0)
        tensor_shapes = [list(np.asarray(output).shape) for output in outputs]
        tensor_dtypes = [str(np.asarray(output).dtype) for output in outputs]
        tensor_finite = [bool(np.isfinite(output).all()) for output in outputs]
        expected_classes = len(classes) if model_kind == "bdd" else 80
        if len(outputs) != 2 or tensor_shapes[0] != [1, 4, 3024] or tensor_shapes[1] != [1, expected_classes, 3024]:
            raise ValueError(f"{label}: unexpected split output contract {tensor_shapes}")
        if not all(tensor_finite):
            raise ValueError(f"{label}: output contains NaN or Inf")

        def pipeline_once(index: int, keep_prediction: bool) -> dict[str, float]:
            sample = samples[index % len(samples)]
            frame = sample["frame"]
            started = time.perf_counter()
            canvas, scale, pad_x, pad_y = letterbox_bgr(frame, (512, 288))
            preprocess_ms = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            outputs = runner.inference(inputs=[canvas[None, ...]], data_format="nhwc")
            inference_ms = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            decoded = decode_outputs(outputs, args.conf, args.iou)
            height, width = frame.shape[:2]
            rows = []
            for detection in decoded:
                class_id = int(detection["class_id"])
                class_name = classes.get(class_id, f"class_{class_id}")
                box = detection["bbox"]
                mapped = [
                    max(0.0, min(float(width), (float(box[0]) - pad_x) / scale)),
                    max(0.0, min(float(height), (float(box[1]) - pad_y) / scale)),
                    max(0.0, min(float(width), (float(box[2]) - pad_x) / scale)),
                    max(0.0, min(float(height), (float(box[3]) - pad_y) / scale)),
                ]
                if class_name in COMMON_CLASSES:
                    rows.append(
                        {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": float(detection["confidence"]),
                            "bbox_xyxy": mapped,
                        }
                    )
            postprocess_ms = (time.perf_counter() - started) * 1000.0

            if keep_prediction and sample["name"] not in predictions:
                predictions[str(sample["name"])] = rows
            return {
                "preprocess_ms": preprocess_ms,
                "npu_ms": inference_ms,
                "postprocess_nms_ms": postprocess_ms,
                "pipeline_ms": preprocess_ms + inference_ms + postprocess_ms,
            }

        for index in range(args.pipeline_warmup):
            pipeline_once(index, keep_prediction=True)
        for index in range(args.pipeline_runs):
            stage = pipeline_once(args.pipeline_warmup + index, keep_prediction=True)
            pipeline_samples.append(stage)
        resources_after_measurement = _resources(process)
        npu_status_after = _read_npu_status()
    finally:
        runner.release()

    class_counts: Counter[str] = Counter()
    for rows in predictions.values():
        class_counts.update(row["class_name"] for row in rows)
    resources_after_measurement["cpu_percent"] = resources_after_measurement.pop("process_cpu_percent")

    return {
        "model": label,
        "path": str(model_path),
        "sha256": _sha256(model_path),
        "source_class_names_by_id": {
            str(key): value for key, value in (BDD_SOURCE_NAMES if model_kind == "bdd" else classes).items()
        },
        "class_names_by_id": {str(key): value for key, value in classes.items()},
        "input": {"shape_nhwc": [1, 288, 512, 3], "dtype": "uint8", "channel_order": "RGB"},
        "outputs": {
            "shapes": tensor_shapes,
            "dtypes": tensor_dtypes,
            "finite": tensor_finite,
        },
        "npu_only": {
            "warmup_runs": args.warmup,
            "measured_runs": len(npu_samples),
            "latency": summarize_latency(npu_samples),
            "fps_from_mean": round(1000.0 / float(np.mean(npu_samples)), 3),
        },
        "pipeline": {
            "warmup_runs": args.pipeline_warmup,
            "measured_runs": len(pipeline_samples),
            "stages": {
                name: summarize_latency([sample[name] for sample in pipeline_samples])
                for name in ("preprocess_ms", "npu_ms", "postprocess_nms_ms", "pipeline_ms")
            },
            "fps_from_mean": round(
                1000.0 / float(np.mean([sample["pipeline_ms"] for sample in pipeline_samples])), 3
            ),
        },
        "detections": {
            "images": len(predictions),
            "conf": args.conf,
            "nms_iou": args.iou,
            "counts_by_class": dict(sorted(class_counts.items())),
            "by_image": predictions,
        },
        "resources": {
            "before_model": resources_before,
            "after_runtime_init": resources_after_init,
            "after_measurement": resources_after_measurement,
            "npu_status_before": npu_status_before,
            "npu_status_after": npu_status_after,
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    for model_path in (args.baseline_model, args.candidate_model):
        if not model_path.is_file():
            raise SystemExit(f"model does not exist: {model_path}")
    if not args.manifest.is_file():
        raise SystemExit(f"manifest does not exist: {args.manifest}")

    samples = _load_samples(args)
    try:
        lite_version = importlib.metadata.version("rknn-toolkit-lite2")
    except importlib.metadata.PackageNotFoundError:
        lite_version = "unknown"

    result = {
        "schema_version": 1,
        "device": "RK3566",
        "runtime": {"rknn_toolkit_lite2": lite_version},
        "test_set": {
            "manifest": str(args.manifest),
            "images": [str(sample["name"]) for sample in samples],
            "count": len(samples),
            "note": "BDD100K holdout100 subset; detection counts are not recall/precision without GT",
        },
        "protocol": {
            "conf": args.conf,
            "nms_iou": args.iou,
            "npu_warmup_runs": args.warmup,
            "npu_measured_runs": args.runs,
            "pipeline_warmup_runs": args.pipeline_warmup,
            "pipeline_measured_runs": args.pipeline_runs,
            "models_run_sequentially": True,
            "pipeline_excludes_disk_decode": True,
        },
        "models": [
            _run_model(
                label=args.baseline_label,
                model_path=args.baseline_model,
                classes=COCO_FOCUS_NAMES if args.baseline_kind == "coco" else BDD_NAMES,
                model_kind=args.baseline_kind,
                samples=samples,
                args=args,
            ),
            _run_model(
                label=args.candidate_label,
                model_path=args.candidate_model,
                classes=BDD_NAMES if args.candidate_kind == "bdd" else COCO_FOCUS_NAMES,
                model_kind=args.candidate_kind,
                samples=samples,
                args=args,
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "models": [m["model"] for m in result["models"]]}, indent=2))


if __name__ == "__main__":
    main()
