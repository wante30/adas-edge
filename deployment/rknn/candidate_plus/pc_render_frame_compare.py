"""Render a same-frame FP32 ONNX vs RK3566 RKNN comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import draw_legend, letterbox_rgb, masks_from_logits, overlay_masks


def panel(image, title):
    image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (text_w, _), _ = cv2.getTextSize(title, font, scale, thickness)
    x = max(18, image.shape[1] - text_w - 18)
    y = 38
    cv2.putText(image, title, (x, y), font, scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, title, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--rknn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import onnxruntime as ort

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    rgb = np.load(str(args.input)).astype(np.uint8)[0]
    _, geometry = letterbox_rgb(bgr)
    nchw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    fp32_logits = session.run(None, {session.get_inputs()[0].name: nchw})[0]
    rknn_logits = np.load(str(args.rknn)).astype(np.float32)
    fp32_masks, _ = masks_from_logits(fp32_logits, geometry, 0.5)
    rknn_masks, _ = masks_from_logits(rknn_logits, geometry, 0.5)

    fp32_overlay = draw_legend(overlay_masks(bgr, fp32_masks, 0.45))
    rknn_overlay = draw_legend(overlay_masks(bgr, rknn_masks, 0.45))
    diff = np.zeros_like(bgr)
    for c in range(4):
        diff[np.logical_xor(fp32_masks[c], rknn_masks[c])] = (0, 255, 255)

    h, w = bgr.shape[:2]
    scale = min(1.0, 960.0 / w)
    size = (int(round(w * scale)), int(round(h * scale)))
    shrink = lambda img: cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    top = np.hstack([panel(shrink(bgr), "original"), panel(shrink(fp32_overlay), "FP32 ONNX")])
    bottom = np.hstack(
        [panel(shrink(rknn_overlay), "RK3566 INT8 QAT"), panel(shrink(diff), "FP32 / INT8 disagreement")]
    )
    canvas = np.vstack([top, bottom])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", args.output)


if __name__ == "__main__":
    main()
