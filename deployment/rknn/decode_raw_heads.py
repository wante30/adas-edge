"""CPU decoder for RKNN YOLOv8 raw detection heads.

The exported raw-head graph emits three tensors per feature scale:
regression DFL logits, class scores, and the objectness/class score sum.
The NPU model is kept unchanged; this module only decodes its outputs.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def _nchw(value: np.ndarray, channels: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4 or array.shape[0] != 1:
        raise ValueError(f"{label} must be [1,C,H,W] or [1,H,W,C], got {array.shape}")
    if array.shape[1] == channels:
        return array
    if array.shape[-1] == channels:
        return array.transpose(0, 3, 1, 2)
    raise ValueError(f"{label} has no channel dimension of size {channels}: {array.shape}")


def looks_like_raw_head_triplets(outputs: Sequence[np.ndarray]) -> bool:
    """Return whether outputs have the raw-head (reg, cls, score) layout."""
    if len(outputs) < 3 or len(outputs) % 3:
        return False
    try:
        for index in range(0, len(outputs), 3):
            box = _nchw(outputs[index], 64, "regression")
            _, _, height, width = box.shape
            scores = np.asarray(outputs[index + 1])
            if scores.ndim == 3:
                scores = scores[None]
            if scores.ndim != 4 or scores.shape[0] != 1:
                return False
            if scores.shape[2:] != (height, width) and scores.shape[1:3] != (height, width):
                return False
            score_sum = np.asarray(outputs[index + 2])
            if score_sum.ndim == 3:
                score_sum = score_sum[None]
            if score_sum.ndim == 4 and score_sum.shape[1:3] == (height, width) and score_sum.shape[-1] == 1:
                score_sum = score_sum.transpose(0, 3, 1, 2)
            if score_sum.shape != (1, 1, height, width):
                return False
    except (TypeError, ValueError):
        return False
    return True


def decode_raw_heads_thresholded(
    outputs: Sequence[np.ndarray],
    *,
    input_width: int,
    input_height: int,
    confidence: float,
    iou: float,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> list[list[float]]:
    """Decode raw heads and return ``[x1,y1,x2,y2,confidence,class_id]`` rows."""
    if not looks_like_raw_head_triplets(outputs):
        raise ValueError(f"expected raw head triplets, got {len(outputs)} outputs")
    if not 0.0 <= confidence <= 1.0 or not 0.0 < iou < 1.0:
        raise ValueError("invalid confidence or NMS threshold")
    if input_width <= 0 or input_height <= 0 or scale <= 0:
        raise ValueError("invalid model input transform")

    by_class: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    weights = np.arange(16, dtype=np.float32)
    for index in range(0, len(outputs), 3):
        box = _nchw(outputs[index], 64, f"scale {index // 3} regression")
        _, _, height, width = box.shape

        scores = np.asarray(outputs[index + 1], dtype=np.float32)
        if scores.ndim == 3:
            scores = scores[None]
        if scores.shape[2:] == (height, width):
            pass
        elif scores.shape[1:3] == (height, width):
            scores = scores.transpose(0, 3, 1, 2)
        else:
            raise ValueError(f"class grid does not match regression grid: {scores.shape}")

        score_sum = np.asarray(outputs[index + 2], dtype=np.float32)
        if score_sum.ndim == 3:
            score_sum = score_sum[None]
        if score_sum.ndim == 4 and score_sum.shape[1:3] == (height, width) and score_sum.shape[-1] == 1:
            score_sum = score_sum.transpose(0, 3, 1, 2)
        if score_sum.shape != (1, 1, height, width):
            raise ValueError(f"invalid score-sum shape: {score_sum.shape}")

        flat_scores = scores[0].reshape(scores.shape[1], -1)
        possible = np.flatnonzero(flat_scores.max(axis=0) >= confidence)
        if not possible.size:
            continue
        possible_scores = flat_scores[:, possible]
        class_ids = possible_scores.argmax(axis=0)
        confidences = possible_scores[class_ids, np.arange(possible.size)]
        accepted = confidences >= confidence
        anchor_ids = possible[accepted]
        if not anchor_ids.size:
            continue
        class_ids = class_ids[accepted]
        confidences = confidences[accepted]

        logits = box[0].reshape(4, 16, -1)[:, :, anchor_ids].transpose(2, 0, 1)
        logits -= logits.max(axis=2, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=2, keepdims=True)
        distances = (probabilities * weights).sum(axis=2)

        grid_x = (anchor_ids % width).astype(np.float32)
        grid_y = (anchor_ids // width).astype(np.float32)
        stride_x, stride_y = input_width / width, input_height / height
        boxes = np.column_stack((
            (grid_x + 0.5 - distances[:, 0]) * stride_x,
            (grid_y + 0.5 - distances[:, 1]) * stride_y,
            (grid_x + 0.5 + distances[:, 2]) * stride_x,
            (grid_y + 0.5 + distances[:, 3]) * stride_y,
        ))
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        valid = (
            np.isfinite(boxes).all(axis=1)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        boxes, class_ids, confidences = boxes[valid], class_ids[valid], confidences[valid]
        for class_id in np.unique(class_ids):
            mask = class_ids == class_id
            by_class.setdefault(int(class_id), []).append((boxes[mask], confidences[mask]))

    result: list[list[float]] = []
    for class_id, rows in by_class.items():
        boxes = np.concatenate([row[0] for row in rows], axis=0)
        confidences = np.concatenate([row[1] for row in rows], axis=0)
        xywh = boxes.copy()
        xywh[:, 2:] -= xywh[:, :2]
        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(), confidences.astype(float).tolist(), 0.0, float(iou)
        )
        if indices is None or len(indices) == 0:
            continue
        for selected in np.asarray(indices).reshape(-1):
            x1, y1, x2, y2 = boxes[int(selected)]
            result.append([
                float(x1), float(y1), float(x2), float(y2),
                float(confidences[int(selected)]), float(class_id),
            ])
    return sorted(result, key=lambda row: row[4], reverse=True)
