"""Shared Candidate-plus road-segmentation preprocessing/postprocessing.

The contract is taken from the training code:
  * source image is BGR (OpenCV), converted to RGB;
  * aspect-preserving letterbox to 320x192, fill RGB(114,114,114);
  * ONNX/RKNN input is RGB, uint8/255, mean 0, std 1;
  * output is logits [1,4,192,320], channel order
    drivable_area, solid_lane, dashed_lane, crosswalk;
  * four independent sigmoid probabilities, threshold >= 0.5.

Restoration follows the training evaluator: sigmoid in the letterboxed
probability domain, crop the padding, bilinear resize to the native frame,
then threshold.  It is deliberately not "threshold first, then resize".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

CLASS_NAMES = ("drivable_area", "solid_lane", "dashed_lane", "crosswalk")
CLASS_NAMES_CN = ("可行驶区域", "实线", "虚线", "斑马线")
CLASS_NAMES_EN = ("Drivable", "Solid lane", "Dashed lane", "Crosswalk")
# BGR colors requested by the task: green, red, cyan, magenta.
CLASS_COLORS_BGR = (
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
)
PAD_VALUE = 114
INPUT_SIZE = (320, 192)  # width, height


def letterbox_rgb(
    bgr: np.ndarray, size: Tuple[int, int] = INPUT_SIZE, pad: int = PAD_VALUE
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Return letterboxed RGB uint8 HWC and the exact geometry used."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected BGR HWC image, got %r" % (bgr.shape,))
    width, height = size
    ih, iw = bgr.shape[:2]
    scale = min(width / float(iw), height / float(ih))
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    left = (width - nw) // 2
    top = (height - nh) // 2
    canvas = np.full((height, width, 3), pad, np.uint8)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    canvas[top : top + nh, left : left + nw] = rgb
    geometry = {
        "native_h": ih,
        "native_w": iw,
        "resized_h": nh,
        "resized_w": nw,
        "top": top,
        "left": left,
    }
    return canvas, geometry


def to_nhwc_uint8(rgb: np.ndarray) -> np.ndarray:
    """Add batch dimension for RKNN Lite ``data_format='nhwc'``."""
    return np.ascontiguousarray(rgb[None, ...], dtype=np.uint8)


def to_nchw_float(rgb: np.ndarray) -> np.ndarray:
    """FP32 ONNX/PyTorch input: RGB / 255, NCHW."""
    x = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return x[None, ...]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def crop_letterbox(prob: np.ndarray, geometry: Dict[str, int]) -> np.ndarray:
    """prob: [C,H,W] in the 320x192 letterboxed grid -> [C,nh,nw]."""
    nh = geometry["resized_h"]
    nw = geometry["resized_w"]
    top = geometry["top"]
    left = geometry["left"]
    return prob[:, top : top + nh, left : left + nw]


def restore_prob(prob: np.ndarray, geometry: Dict[str, int]) -> np.ndarray:
    """Crop padding, bilinear resize probabilities to native resolution."""
    cropped = crop_letterbox(prob, geometry)
    ih = geometry["native_h"]
    iw = geometry["native_w"]
    # One HWC resize is much cheaper than four per-channel calls on RK3566.
    hwc = np.ascontiguousarray(cropped.transpose(1, 2, 0))
    restored = cv2.resize(hwc, (iw, ih), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(restored.transpose(2, 0, 1))


def masks_from_logits(
    logits: np.ndarray, geometry: Dict[str, int], threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (native masks [C,H,W] bool, native probabilities [C,H,W] float32)."""
    prob = sigmoid(np.asarray(logits, dtype=np.float32))
    if prob.ndim == 4:
        prob = prob[0]
    restored = restore_prob(prob, geometry)
    return restored >= threshold, restored


def overlay_masks(
    bgr: np.ndarray, masks: Sequence[np.ndarray], alpha: float = 0.45
) -> np.ndarray:
    """Blend class masks in the requested green/red/cyan/magenta colors.

    A single indexed label map plus one alpha blend keeps the native-resolution
    mask boundaries while avoiding four full-frame float conversions.  When
    masks overlap, the later class (crosswalk) has display priority; the four
    model channels remain independent and are not merged before thresholding.
    """
    if len(masks) != len(CLASS_COLORS_BGR):
        raise ValueError("expected four masks")
    h, w = bgr.shape[:2]
    color_map = np.zeros_like(bgr)
    for mask, color in zip(masks, CLASS_COLORS_BGR):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (h, w):
            raise ValueError("mask shape %r != image shape %r" % (mask.shape, (h, w)))
        # Limit the expensive boolean assignment to the mask bounding box.
        x, y, bw, bh = cv2.boundingRect(mask.astype(np.uint8))
        if bw and bh:
            roi = color_map[y : y + bh, x : x + bw]
            roi[mask[y : y + bh, x : x + bw]] = color
    return cv2.addWeighted(bgr, 1.0 - alpha, color_map, alpha, 0.0)


def draw_legend(bgr: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """Draw a compact class-color legend in the top-left corner."""
    out = bgr.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.35, min(0.9, w / 1280.0 * 0.7))
    thickness = max(1, int(round(scale * 2)))
    line_h = int(round(28 * scale))
    box_w = int(round(260 * scale))
    box_h = line_h * 4 + int(round(12 * scale))
    x0, y0 = 8, 8
    x1, y1 = min(w, x0 + box_w), min(h, y0 + box_h)
    roi = out[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x1 - x0 - 1, y1 - y0 - 1), (0, 0, 0), -1)
    out[y0:y1, x0:x1] = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
    for i, (name, color) in enumerate(zip(CLASS_NAMES_EN, CLASS_COLORS_BGR)):
        y = y0 + line_h * i + int(round(20 * scale))
        cv2.rectangle(
            out,
            (x0 + int(10 * scale), y - int(12 * scale)),
            (x0 + int(28 * scale), y - int(2 * scale)),
            color,
            -1,
        )
        cv2.putText(
            out,
            name,
            (x0 + int(38 * scale), y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return out


def load_manifest(path: Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
