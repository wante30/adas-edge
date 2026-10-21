"""Collect BDD predictions from Rockchip Model Zoo optimized YOLOv6/YOLOv8 ONNX graphs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.profile_rknn_single_stream import letterbox_bgr

TARGET_COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
}


def prepare_model_zoo_onnx_input(
    frame_bgr: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, float, int, int]:
    """Prepare the Rockchip Model Zoo ONNX input and letterbox metadata."""
    canvas_rgb, scale, pad_x, pad_y = letterbox_bgr(frame_bgr, size, pad_value=0)
    tensor = canvas_rgb.transpose(2, 0, 1)[None, ...].astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor), scale, pad_x, pad_y


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-family", choices=("yolov6", "yolov8"), required=True)
    parser.add_argument("--bdd-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args(argv)

    if args.width < 32 or args.height < 32 or args.width % 32 or args.height % 32:
        parser.error("canvas width and height must be at least 32 and divisible by 32")
    if args.limit < 1 or args.max_det < 1:
        parser.error("limit and max-det must be positive")
    if not 0.0 <= args.score_floor <= 1.0 or not 0.0 <= args.iou <= 1.0:
        parser.error("score-floor and iou must be in [0, 1]")
    return args


def decode_model_zoo_outputs(
    outputs: Sequence[np.ndarray],
    model_family: str,
    input_shape: tuple[int, int],
    confidence_threshold: float = 0.001,
    iou_threshold: float = 0.45,
    max_det: int = 100,
) -> list[tuple[int, float, float, float, float, float]]:
    """Decode the official 9-output layout; input_shape is (height, width)."""
    if model_family not in {"yolov6", "yolov8"}:
        raise ValueError(f"unsupported model family: {model_family}")
    if len(outputs) != 9:
        raise ValueError(f"expected 9 outputs across 3 scales, got {len(outputs)} outputs")

    from deployment.rknn.decode_yolo_output import _nms

    input_height, input_width = input_shape
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []
    for branch in range(3):
        position = np.asarray(outputs[branch * 3], dtype=np.float32)
        class_scores = np.asarray(outputs[branch * 3 + 1], dtype=np.float32)
        score_sum = np.asarray(outputs[branch * 3 + 2], dtype=np.float32)
        expected_channels = 4 if model_family == "yolov6" else 64
        if position.ndim != 4 or position.shape[:2] != (1, expected_channels):
            raise ValueError(f"expected position [1,{expected_channels},H,W], got {position.shape}")
        height, width = position.shape[2:]
        if class_scores.shape != (1, 80, height, width) or score_sum.shape != (1, 1, height, width):
            raise ValueError(f"unexpected score output shapes {class_scores.shape} and {score_sum.shape}")
        if not np.isfinite(position).all() or not np.isfinite(class_scores).all():
            raise ValueError("model outputs contain non-finite values")

        if model_family == "yolov6":
            distances = position[0]
        else:
            distribution = position[0].reshape(4, 16, height, width)
            distribution -= distribution.max(axis=1, keepdims=True)
            probabilities = np.exp(distribution)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            distances = (probabilities * np.arange(16, dtype=np.float32)[None, :, None, None]).sum(axis=1)

        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        stride_x, stride_y = input_width / width, input_height / height
        x1 = (grid_x + 0.5 - distances[0]) * stride_x
        y1 = (grid_y + 0.5 - distances[1]) * stride_y
        x2 = (grid_x + 0.5 + distances[2]) * stride_x
        y2 = (grid_y + 0.5 + distances[3]) * stride_y
        boxes = np.column_stack((x1.reshape(-1), y1.reshape(-1), x2.reshape(-1), y2.reshape(-1)))
        probabilities = class_scores[0].transpose(1, 2, 0).reshape(-1, 80)
        classes = probabilities.argmax(axis=1).astype(np.int32)
        scores = probabilities[np.arange(probabilities.shape[0]), classes]
        valid = scores >= confidence_threshold
        all_boxes.append(boxes[valid])
        all_scores.append(scores[valid])
        all_classes.append(classes[valid])

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_classes)
    kept: list[int] = []
    for class_id in np.unique(classes):
        indices = np.flatnonzero(classes == class_id)
        kept.extend(indices[_nms(boxes[indices], scores[indices], iou_threshold)].tolist())
    kept = sorted(kept, key=lambda index: float(scores[index]), reverse=True)[:max_det]
    return [
        (int(classes[index]), *(float(value) for value in boxes[index]), float(scores[index]))
        for index in kept
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.model.is_file():
        raise SystemExit(f"model does not exist: {args.model}")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite prediction dump: {args.out}")
    image_root = args.bdd_root / "images" / "100k" / "val"
    image_paths = sorted(image_root.glob("*.jpg"))[: args.limit]
    if not image_paths:
        raise SystemExit(f"no BDD images under {image_root}")

    import cv2
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1:
        raise SystemExit(f"expected one ONNX input, got {len(session.get_inputs())}")
    input_info = session.get_inputs()[0]
    expected_shape = [1, 3, args.height, args.width]
    if input_info.shape != expected_shape:
        raise SystemExit(f"ONNX input shape {input_info.shape} does not match {expected_shape}")

    per_image: list[dict[str, object]] = []
    started = time.perf_counter()
    for path in image_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"cannot decode BDD image: {path}")
        tensor, scale, pad_x, pad_y = prepare_model_zoo_onnx_input(frame, (args.width, args.height))
        outputs = session.run(None, {input_info.name: tensor})
        rows = decode_model_zoo_outputs(
            outputs,
            args.model_family,
            (args.height, args.width),
            args.score_floor,
            args.iou,
            args.max_det,
        )
        source_height, source_width = frame.shape[:2]
        detections: list[list[float | int]] = []
        for class_id, x1, y1, x2, y2, score in rows:
            if class_id not in TARGET_COCO_NAMES:
                continue
            x1 = float(np.clip((x1 - pad_x) / scale, 0, source_width))
            x2 = float(np.clip((x2 - pad_x) / scale, 0, source_width))
            y1 = float(np.clip((y1 - pad_y) / scale, 0, source_height))
            y2 = float(np.clip((y2 - pad_y) / scale, 0, source_height))
            if x2 > x1 and y2 > y1:
                detections.append([class_id, x1, y1, x2, y2, score])
        per_image.append({"image": path.name, "detections": detections})
    elapsed = time.perf_counter() - started

    payload = {
        "model": str(args.model),
        "model_family": args.model_family,
        "width": args.width,
        "height": args.height,
        "score_floor": args.score_floor,
        "nms_iou": args.iou,
        "max_det": args.max_det,
        "observed_input_shapes": [expected_shape],
        "model_names": {str(index): name for index, name in TARGET_COCO_NAMES.items()},
        "per_image": per_image,
        "runtime": {
            "device": "CPUExecutionProvider",
            "wall_ms_per_image": elapsed * 1000.0 / len(image_paths),
            "wall_total_s": elapsed,
            "timing_scope": "PC ONNX Runtime; includes image decode, letterbox, forward, decode, and NMS",
            "inference_sample_count": len(image_paths),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    print(
        json.dumps(
            {
                "images": len(image_paths),
                "model_family": args.model_family,
                "input_shape": expected_shape,
                "out": str(args.out),
                **payload["runtime"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
