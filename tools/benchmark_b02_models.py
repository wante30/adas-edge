"""Measure cached BDD frames with the backends available on this PC."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def measure(
    operation: Callable[[int], None],
    synchronize: Callable[[], None],
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    for index in range(warmup):
        operation(index)
    synchronize()
    samples_ms = []
    for index in range(iterations):
        synchronize()
        started = time.perf_counter_ns()
        operation(index)
        synchronize()
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    average_ms = sum(samples_ms) / len(samples_ms)
    return {
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "mean_ms": average_ms,
        "fps": 1000.0 / average_ms,
    }


def load_frames(bdd_root: Path, prediction_dump: Path) -> list[tuple[str, object]]:
    import cv2

    payload = json.loads(prediction_dump.read_text(encoding="utf-8"))
    image_root = bdd_root / "images" / "100k" / "val"
    frames = []
    for entry in payload["per_image"]:
        path = image_root / entry["image"]
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"cannot read BDD frame: {path}")
        frames.append((entry["image"], frame))
    if not frames:
        raise ValueError("prediction dump contains no frames")
    return frames


def benchmark_yolov6_cpu(
    model_path: Path,
    frames: list[tuple[str, object]],
    warmup: int,
    iterations: int,
) -> dict:
    import numpy as np
    import onnxruntime as ort

    from tools.collect_rknn_zoo_onnx_predictions import (
        decode_model_zoo_outputs,
        prepare_model_zoo_onnx_input,
        TARGET_COCO_NAMES,
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    expected = [1, 3, 288, 512]
    if input_info.shape != expected:
        raise ValueError(f"expected YOLOv6 input {expected}, got {input_info.shape}")
    tensors = [prepare_model_zoo_onnx_input(frame, (512, 288)) for _, frame in frames]

    def inference(index: int) -> None:
        session.run(None, {input_info.name: tensors[index % len(tensors)][0]})

    def pipeline(index: int) -> None:
        frame = frames[index % len(frames)][1]
        tensor, scale, pad_x, pad_y = prepare_model_zoo_onnx_input(frame, (512, 288))
        outputs = session.run(None, {input_info.name: tensor})
        rows = decode_model_zoo_outputs(outputs, "yolov6", (288, 512), 0.25, 0.45, 100)
        height, width = frame.shape[:2]
        mapped_boxes = []
        for class_id, x1, y1, x2, y2, _score in rows:
            if class_id in TARGET_COCO_NAMES:
                mapped = (
                    float(np.clip((x1 - pad_x) / scale, 0, width)),
                    float(np.clip((y1 - pad_y) / scale, 0, height)),
                    float(np.clip((x2 - pad_x) / scale, 0, width)),
                    float(np.clip((y2 - pad_y) / scale, 0, height)),
                )
                if mapped[2] > mapped[0] and mapped[3] > mapped[1]:
                    mapped_boxes.append((class_id, *mapped))

    no_sync = lambda: None
    return {
        "backend": "ONNX Runtime CPUExecutionProvider",
        "reason": "CUDAExecutionProvider is not installed in the available Windows ONNX Runtime; WSL has no ONNX Runtime",
        "images": len(frames),
        "pure_inference": measure(inference, no_sync, warmup, iterations),
        "full_pipeline": measure(pipeline, no_sync, warmup, iterations),
    }


def benchmark_ultralytics_cuda(
    weights: Path,
    frames: list[tuple[str, object]],
    warmup: int,
    iterations: int,
) -> dict:
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")
    model = YOLO(str(weights))
    predict_args = {
        "imgsz": (288, 512),
        "conf": 0.25,
        "iou": 0.45,
        "max_det": 100,
        "device": 0,
        "half": False,
        "rect": False,
        "verbose": False,
    }
    model.predict(source=frames[0][1], **predict_args)
    predictor = model.predictor
    tensors = [predictor.preprocess([frame]) for _, frame in frames]

    def inference(index: int) -> None:
        with torch.inference_mode():
            predictor.model(tensors[index % len(tensors)])

    def pipeline(index: int) -> None:
        frame = frames[index % len(frames)][1]
        with torch.inference_mode():
            batch = predictor.preprocess([frame])
            predictions = predictor.inference(batch)
            predictor.postprocess(predictions, batch, [frame])

    synchronize = torch.cuda.synchronize
    return {
        "backend": "Ultralytics PyTorch CUDA FP32",
        "device": torch.cuda.get_device_name(0),
        "images": len(frames),
        "pure_inference": measure(inference, synchronize, warmup, iterations),
        "full_pipeline": measure(pipeline, synchronize, warmup, iterations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdd-root", type=Path, required=True)
    parser.add_argument("--sample-dump", type=Path, required=True)
    parser.add_argument("--yolov6-onnx", type=Path, required=True)
    parser.add_argument("--yolov8-weights", type=Path, required=True)
    parser.add_argument("--yolo11-weights", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be nonnegative and iterations must be positive")
    frames = load_frames(args.bdd_root, args.sample_dump)
    report = {
        "input": {"width": 512, "height": 288, "batch": 1, "images": len(frames)},
        "yolov6n": benchmark_yolov6_cpu(args.yolov6_onnx, frames, args.warmup, args.iterations),
        "yolov8n": benchmark_ultralytics_cuda(args.yolov8_weights, frames, args.warmup, args.iterations),
        "yolo11n": benchmark_ultralytics_cuda(args.yolo11_weights, frames, args.warmup, args.iterations),
        "comparison_note": "YOLOv6n CPU latency is not comparable with the two CUDA rows.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
