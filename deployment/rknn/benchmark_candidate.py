"""Benchmark candidate RKNN models on-device with optional YOLOv8 split postprocessing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-npy", type=Path, required=True, help="HWC or NHWC uint8 tensor, already letterboxed")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--image", type=Path, help="source BGR image for the full pipeline measurement")
    parser.add_argument("--decoder", choices=("yolov8-split",), help="decoder for [1,4,N] boxes + [1,C,N] probabilities")
    parser.add_argument("--pipeline-warmup", type=int, default=5)
    parser.add_argument("--pipeline-runs", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)

    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be non-negative and --runs must be positive")
    if args.pipeline_warmup < 0 or args.pipeline_runs < 1:
        parser.error("--pipeline-warmup must be non-negative and --pipeline-runs must be positive")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        parser.error("--conf and --iou must be in [0, 1]")
    if bool(args.image) != bool(args.decoder):
        parser.error("--image and a supported --decoder must be supplied together")
    return args


def load_input_tensor(path: Path) -> np.ndarray:
    tensor = np.load(path, allow_pickle=False)
    if tensor.ndim == 3:
        tensor = tensor[None, ...]
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[-1] != 3 or tensor.dtype != np.uint8:
        raise ValueError(f"expected HWC or 1xHxWx3 uint8 input, got shape={tensor.shape}, dtype={tensor.dtype}")
    return np.ascontiguousarray(tensor)


def summarize_latency(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("latency samples must be a non-empty finite sequence")
    return {
        "mean_ms": round(float(array.mean()), 3),
        "p50_ms": round(float(np.percentile(array, 50)), 3),
        "p95_ms": round(float(np.percentile(array, 95)), 3),
        "min_ms": round(float(array.min()), 3),
        "max_ms": round(float(array.max()), 3),
    }


def decode_yolov8_split(
    outputs: Sequence[np.ndarray], confidence_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(outputs) != 2:
        raise ValueError(f"expected two split outputs, got {len(outputs)}")
    boxes, scores = (np.asarray(output, dtype=np.float32) for output in outputs)
    if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[1] != 4:
        raise ValueError(f"expected boxes [1,4,N], got {boxes.shape}")
    if scores.ndim != 3 or scores.shape[0] != 1 or scores.shape[2] != boxes.shape[2]:
        raise ValueError(f"expected scores [1,C,{boxes.shape[2]}], got {scores.shape}")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("model outputs contain non-finite values")

    xywh = boxes[0].T
    xyxy = np.column_stack(
        (
            xywh[:, 0] - xywh[:, 2] / 2,
            xywh[:, 1] - xywh[:, 3] / 2,
            xywh[:, 0] + xywh[:, 2] / 2,
            xywh[:, 1] + xywh[:, 3] / 2,
        )
    )
    class_scores = scores[0].T
    class_ids = class_scores.argmax(axis=1)
    confidence = class_scores[np.arange(class_scores.shape[0]), class_ids]
    valid = confidence >= confidence_threshold
    return xyxy[valid], confidence[valid], class_ids[valid].astype(np.int32)


def classwise_nms(
    boxes: np.ndarray, confidence: np.ndarray, class_ids: np.ndarray, iou_threshold: float
) -> list[int]:
    from tools.profile_rknn_single_stream import _nms

    keep: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        local_keep = _nms(boxes[indices], confidence[indices], iou_threshold)
        keep.extend(indices[local_keep].tolist())
    return sorted(keep, key=lambda index: float(confidence[index]), reverse=True)


def read_npu_frequencies(sys_root: Path = Path("/sys")) -> dict[str, int]:
    values: dict[str, int] = {}
    devfreq = sys_root / "class" / "devfreq"
    if not devfreq.is_dir():
        return values
    for node in sorted(devfreq.iterdir()):
        if "npu" not in node.name.lower():
            continue
        try:
            values[node.name] = int((node / "cur_freq").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
    return values


def read_npu_load(sys_root: Path = Path("/sys")) -> str | None:
    path = sys_root / "kernel" / "debug" / "rknpu" / "load"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.model.is_file():
        raise SystemExit(f"model does not exist: {args.model}")
    if not args.input_npy.is_file():
        raise SystemExit(f"input tensor does not exist: {args.input_npy}")
    if args.image and not args.image.is_file():
        raise SystemExit(f"source image does not exist: {args.image}")
    if args.json_output and args.json_output.exists():
        raise SystemExit(f"refusing to overwrite result: {args.json_output}")

    input_tensor = load_input_tensor(args.input_npy)
    try:
        import psutil

        process = psutil.Process(os.getpid())
        process.cpu_percent(interval=None)
    except ImportError:
        psutil = None
        process = None

    if psutil:
        psutil.cpu_percent(interval=0.2)
        process.cpu_percent(interval=None)

    def snapshot() -> dict[str, object]:
        return {
            "system_cpu_percent": psutil.cpu_percent(interval=None) if psutil else None,
            "process_cpu_percent": process.cpu_percent(interval=None) if process else None,
            "process_rss_bytes": process.memory_info().rss if process else None,
            "npu_frequency_hz": read_npu_frequencies(),
            "npu_load": read_npu_load(),
        }

    from rknnlite.api import RKNNLite

    runner = RKNNLite(verbose=False)
    npu_timings: list[float] = []
    pipeline_samples: list[dict[str, float]] = []
    output_tensors: list[np.ndarray] = []
    rss_after_init = None
    pipeline_detections = 0
    resources_before: dict[str, object] = {}
    resources_after_npu: dict[str, object] = {}
    resources_after_pipeline: dict[str, object] | None = None
    try:
        if runner.load_rknn(str(args.model)) != 0:
            raise RuntimeError(f"load_rknn failed: {args.model}")
        if runner.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
            raise RuntimeError(f"init_runtime failed: {args.model}")
        rss_after_init = process.memory_info().rss if process else None
        resources_before = snapshot()

        for _ in range(args.warmup):
            runner.inference(inputs=[input_tensor], data_format="nhwc")
        for _ in range(args.runs):
            started = time.perf_counter()
            output_tensors = runner.inference(inputs=[input_tensor], data_format="nhwc")
            npu_timings.append((time.perf_counter() - started) * 1000.0)
        rss_after_npu = process.memory_info().rss if process else None
        resources_after_npu = snapshot()

        if args.image:
            import cv2

            from tools.profile_rknn_single_stream import letterbox_bgr

            input_height, input_width = input_tensor.shape[1:3]

            def pipeline_once() -> tuple[dict[str, float], int]:
                started = time.perf_counter()
                frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
                source_decode_ms = (time.perf_counter() - started) * 1000.0
                if frame is None:
                    raise RuntimeError(f"cannot decode source image: {args.image}")

                started = time.perf_counter()
                canvas, _, _, _ = letterbox_bgr(frame, (input_width, input_height))
                preprocess_ms = (time.perf_counter() - started) * 1000.0

                started = time.perf_counter()
                outputs = runner.inference(inputs=[canvas[None, ...]], data_format="nhwc")
                npu_ms = (time.perf_counter() - started) * 1000.0

                started = time.perf_counter()
                boxes, confidence, class_ids = decode_yolov8_split(outputs, args.conf)
                output_decode_ms = (time.perf_counter() - started) * 1000.0

                started = time.perf_counter()
                keep = classwise_nms(boxes, confidence, class_ids, args.iou)
                nms_ms = (time.perf_counter() - started) * 1000.0
                return (
                    {
                        "source_decode": source_decode_ms,
                        "preprocess": preprocess_ms,
                        "npu": npu_ms,
                        "output_decode": output_decode_ms,
                        "nms": nms_ms,
                    },
                    len(keep),
                )

            for _ in range(args.pipeline_warmup):
                pipeline_once()
            for _ in range(args.pipeline_runs):
                stages, count = pipeline_once()
                stages["end_to_end"] = sum(stages.values())
                pipeline_samples.append(stages)
                pipeline_detections += count
            resources_after_pipeline = snapshot()
    finally:
        runner.release()

    npu_summary = summarize_latency(npu_timings)
    output_summary = [
        {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(np.isfinite(output).all()),
        }
        for output in output_tensors
    ]
    try:
        lite_version = importlib.metadata.version("rknn-toolkit-lite2")
    except importlib.metadata.PackageNotFoundError:
        lite_version = "unknown"
    driver_version_path = Path("/sys/kernel/debug/rknpu/version")
    try:
        driver_version = driver_version_path.read_text(encoding="utf-8").strip()
    except OSError:
        driver_version = None

    report: dict[str, object] = {
        "schema_version": 1,
        "model": str(args.model),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "input_tensor": {
            "path": str(args.input_npy),
            "shape_nhwc": list(input_tensor.shape),
            "dtype": str(input_tensor.dtype),
            "already_letterboxed": True,
        },
        "npu_only": {
            "warmup_runs": args.warmup,
            "measured_runs": len(npu_timings),
            "latency": npu_summary,
            "fps_from_mean": round(1000.0 / npu_summary["mean_ms"], 3),
        },
        "pipeline": None,
        "output_tensors": output_summary,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "rknn_toolkit_lite2": lite_version,
            "rknpu_driver_version": driver_version,
        },
        "resources": {
            "rss_after_init_bytes": rss_after_init,
            "rss_after_npu_bytes": rss_after_npu,
            "before_measurement": resources_before,
            "after_npu_only": resources_after_npu,
            "after_pipeline": resources_after_pipeline or snapshot(),
        },
    }

    if pipeline_samples:
        pipeline_report: dict[str, object] = {
            "source_image": str(args.image),
            "decoder": args.decoder,
            "warmup_runs": args.pipeline_warmup,
            "measured_runs": len(pipeline_samples),
            "stages": {
                stage: summarize_latency([sample[stage] for sample in pipeline_samples])
                for stage in pipeline_samples[0]
            },
            "fps_from_end_to_end_mean": round(
                1000.0 / float(np.mean([sample["end_to_end"] for sample in pipeline_samples])), 3
            ),
            "detections_per_frame": round(pipeline_detections / len(pipeline_samples), 3),
            "note": "same source image is decoded once per measured pipeline iteration",
        }
        report["pipeline"] = pipeline_report

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
