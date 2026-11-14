"""Run a single image through FP32 ONNX and write a color overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import draw_legend, letterbox_rgb, masks_from_logits, overlay_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import onnxruntime as ort

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    rgb, geometry = letterbox_rgb(bgr)
    nchw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    logits = session.run(None, {session.get_inputs()[0].name: nchw})[0]
    masks, _ = masks_from_logits(logits, geometry, 0.5)
    overlay = draw_legend(overlay_masks(bgr, masks, 0.45))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", args.output)


if __name__ == "__main__":
    main()
