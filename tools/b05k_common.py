"""Shared preprocessing, YOLO11 decoding, and one-to-one detection metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameTransform:
    frame_width: int
    frame_height: int
    roi_y0: int
    roi_height: int
    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int

    def network_to_frame(self, box: list[float] | tuple[float, ...]) -> list[float]:
        x1, y1, x2, y2 = map(float, box)
        return [
            min(float(self.frame_width), max(0.0, (x1 - self.pad_x) / self.scale_x)),
            min(float(self.frame_height), max(0.0, (y1 - self.pad_y) / self.scale_y + self.roi_y0)),
            min(float(self.frame_width), max(0.0, (x2 - self.pad_x) / self.scale_x)),
            min(float(self.frame_height), max(0.0, (y2 - self.pad_y) / self.scale_y + self.roi_y0)),
        ]


def prepare_input(
    frame: np.ndarray,
    *,
    input_width: int = 512,
    input_height: int = 192,
    roi_top: float = 0.60,
    preprocess: str = "letterbox",
) -> tuple[np.ndarray, FrameTransform]:
    """Build a normalized RGB/NCHW tensor from the top-of-frame ROI."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image with three channels")
    if input_width <= 0 or input_height <= 0 or not 0 < roi_top <= 1:
        raise ValueError("input dimensions must be positive and roi_top must be in (0, 1]")
    if preprocess not in {"letterbox", "resize"}:
        raise ValueError(f"unsupported preprocess mode: {preprocess}")

    frame_height, frame_width = frame.shape[:2]
    roi_height = max(1, min(frame_height, int(round(frame_height * roi_top))))
    roi = frame[:roi_height]
    if preprocess == "letterbox":
        scale = min(input_width / frame_width, input_height / roi_height)
        resized_width = max(1, round(frame_width * scale))
        resized_height = max(1, round(roi_height * scale))
        resized = cv2.resize(roi, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (input_width - resized_width) // 2
        pad_y = (input_height - resized_height) // 2
        canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
        scale_x = scale_y = scale
    else:
        canvas = cv2.resize(roi, (input_width, input_height), interpolation=cv2.INTER_AREA)
        pad_x = pad_y = 0
        scale_x = input_width / frame_width
        scale_y = input_height / roi_height

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    transform = FrameTransform(
        frame_width=frame_width,
        frame_height=frame_height,
        roi_y0=0,
        roi_height=roi_height,
        scale_x=scale_x,
        scale_y=scale_y,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    return tensor, transform


def box_iou(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float, limit: int) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size and len(keep) < limit:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        overlaps = np.array([box_iou(boxes[current], boxes[index]) for index in rest])
        order = rest[overlaps <= threshold]
    return keep


def decode_yolo11(
    output: Any,
    *,
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 300,
) -> list[dict[str, Any]]:
    """Decode [1, 5, N] xywh + already-sigmoided single-class scores."""
    if isinstance(output, (tuple, list)):
        output = output[0]
    if hasattr(output, "detach"):
        output = output.detach().float().cpu().numpy()
    raw = np.asarray(output, dtype=np.float32)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 2 or raw.shape[0] != 5:
        raise ValueError(f"expected single-class YOLO11 output [1,5,N] or [5,N], got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("YOLO11 output contains non-finite values")

    scores = raw[4]
    if scores.min(initial=0.0) < -1e-4 or scores.max(initial=0.0) > 1.0001:
        raise ValueError("YOLO11 class output is outside [0,1]; expected exported probabilities")
    candidate_ids = np.flatnonzero(scores >= conf)
    if candidate_ids.size == 0:
        return []

    cx, cy, width, height = raw[:4, candidate_ids]
    boxes = np.stack((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2), axis=1)
    valid = (width > 0) & (height > 0)
    boxes = boxes[valid]
    selected_scores = scores[candidate_ids][valid]
    if not len(boxes):
        return []
    keep = _nms(boxes, selected_scores, iou, max_det)
    return [
        {"bbox": boxes[index].astype(float).tolist(), "confidence": float(selected_scores[index])}
        for index in keep
    ]


def size_group(width_px: float) -> str:
    if width_px < 8:
        return "<8"
    if width_px < 16:
        return "8-16"
    if width_px < 32:
        return "16-32"
    return ">32"


def _match_image(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    conf: float,
    iou: float,
) -> tuple[int, int, int, list[int]]:
    ordered = sorted(
        (prediction for prediction in predictions if float(prediction["confidence"]) >= conf),
        key=lambda prediction: -float(prediction["confidence"]),
    )
    used_gt: set[int] = set()
    matched_gt: list[int] = []
    tp = fp = 0
    for prediction in ordered:
        best_index = None
        best_overlap = 0.0
        for index, target in enumerate(ground_truth):
            if index in used_gt:
                continue
            overlap = box_iou(prediction["bbox"], target["bbox"])
            if overlap > best_overlap:
                best_overlap, best_index = overlap, index
        if best_index is not None and best_overlap >= iou:
            used_gt.add(best_index)
            matched_gt.append(best_index)
            tp += 1
        else:
            fp += 1
    return tp, fp, len(ground_truth) - tp, matched_gt


def _average_precision(
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    *,
    floor: float,
    iou: float,
) -> float:
    total_gt = sum(len(items) for items in ground_truth.values())
    if total_gt == 0:
        return 0.0
    ranked = [
        (float(prediction["confidence"]), image_key, prediction["bbox"], order)
        for image_key, items in predictions.items()
        for order, prediction in enumerate(items)
        if float(prediction["confidence"]) >= floor
    ]
    ranked.sort(key=lambda row: (-row[0], row[1], row[3]))
    used: dict[str, set[int]] = {}
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    for _, image_key, bbox, _ in ranked:
        targets = ground_truth.get(image_key, [])
        image_used = used.setdefault(image_key, set())
        best_index = None
        best_overlap = 0.0
        for index, target in enumerate(targets):
            if index in image_used:
                continue
            overlap = box_iou(bbox, target["bbox"])
            if overlap > best_overlap:
                best_overlap, best_index = overlap, index
        if best_index is not None and best_overlap >= iou:
            image_used.add(best_index)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
    if not tp_flags:
        return 0.0
    tp_cumulative = np.cumsum(tp_flags, dtype=np.float64)
    fp_cumulative = np.cumsum(fp_flags, dtype=np.float64)
    recall = tp_cumulative / total_gt
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1.0)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changes = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))


def evaluate_predictions(
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    *,
    conf: float = 0.25,
    iou: float = 0.5,
    ap_floor: float = 0.001,
) -> dict[str, Any]:
    """Compute global AP50 and a fixed operating point with one-to-one matches."""
    keys = sorted(set(ground_truth) | set(predictions))
    groups = {name: {"gt": 0, "tp": 0, "fn": 0} for name in ("<8", "8-16", "16-32", ">32")}
    tp = fp = fn = 0
    for image_key in keys:
        targets = ground_truth.get(image_key, [])
        image_tp, image_fp, image_fn, matched = _match_image(
            targets, predictions.get(image_key, []), conf=conf, iou=iou
        )
        tp += image_tp
        fp += image_fp
        fn += image_fn
        matched_set = set(matched)
        for index, target in enumerate(targets):
            group = size_group(float(target.get("width_px", target["bbox"][2] - target["bbox"][0])))
            groups[group]["gt"] += 1
            if index in matched_set:
                groups[group]["tp"] += 1
            else:
                groups[group]["fn"] += 1

    return {
        "ap50": _average_precision(ground_truth, predictions, floor=ap_floor, iou=iou),
        "ap_conf_floor": ap_floor,
        "operating_point": {"conf": conf, "iou": iou},
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "size_groups": groups,
    }


def should_run_inference(frame_index: int, source_fps: float, schedule: str) -> bool:
    if schedule == "every-frame":
        return True
    if schedule != "1hz":
        raise ValueError(f"unsupported schedule: {schedule}")
    return frame_index % max(1, int(round(source_fps))) == 0
