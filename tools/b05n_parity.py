"""B-05N PyTorch/ONNX parity for the ten-class P2 model (traffic-light decode)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.b05k_common import box_iou  # noqa: E402
from tools.b05n_tl_eval import decode_class, prepare_full  # noqa: E402

DEFAULT_IMAGES = Path(r"D:\Datasets\adas_edge_training\b03c_rknn_validation_20261021\board_holdout50")


def numpy_output(output: Any) -> np.ndarray:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if hasattr(output, "detach"):
        output = output.detach().float().cpu().numpy()
    return np.asarray(output, dtype=np.float32)


def match(left: list[dict], right: list[dict], threshold: float = 0.5):
    remaining = set(range(len(right)))
    pairs = []
    for a in sorted(left, key=lambda x: -x["confidence"]):
        choices = [(box_iou(a["bbox"], right[i]["bbox"]), i) for i in remaining]
        if not choices:
            continue
        overlap, index = max(choices)
        if overlap >= threshold:
            remaining.remove(index)
            pairs.append((a, right[index], overlap))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt-model", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from ultralytics import YOLO
    import onnxruntime as ort
    import torch

    torch_model = YOLO(str(args.pt_model)).model.to(args.device).eval()
    session = ort.InferenceSession(str(args.onnx_model), providers=["CPUExecutionProvider"])
    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    image_paths = sorted(args.images.glob("*.jpg"))[:args.limit]
    max_abs = max_mean = 0.0
    max_frame = None
    ious, coord_errors, conf_errors = [], [], []
    unmatched_pt = unmatched_onnx = 0
    records = []
    for path in image_paths:
        frame = cv2.imread(str(path))
        tensor, transform = prepare_full(frame, 512, 288)
        input_tensor = torch.from_numpy(tensor).to(args.device)
        with torch.inference_mode():
            pt_raw = numpy_output(torch_model(input_tensor))
        onnx_raw = numpy_output(session.run([out.name], {inp.name: tensor})[0])
        if pt_raw.shape != onnx_raw.shape:
            raise RuntimeError((pt_raw.shape, onnx_raw.shape))
        diff = np.abs(pt_raw - onnx_raw)
        frame_max, frame_mean = float(diff.max(initial=0.0)), float(diff.mean())
        if frame_max > max_abs:
            max_abs, max_mean, max_frame = frame_max, frame_mean, path.name
        pt_boxes = decode_class(pt_raw, 8, 0.05, 0.5)
        onnx_boxes = decode_class(onnx_raw, 8, 0.05, 0.5)
        for item in pt_boxes + onnx_boxes:
            x1, y1, x2, y2 = item["bbox"]
            item["bbox"] = [x1 / transform["scale_x"], y1 / transform["scale_y"], x2 / transform["scale_x"], y2 / transform["scale_y"]]
        pairs = match(pt_boxes, onnx_boxes)
        for a, b, overlap in pairs:
            ious.append(overlap)
            coord_errors.append(float(np.mean(np.abs(np.asarray(a["bbox"]) - np.asarray(b["bbox"])))))
            conf_errors.append(abs(a["confidence"] - b["confidence"]))
        unmatched_pt += len(pt_boxes) - len(pairs)
        unmatched_onnx += len(onnx_boxes) - len(pairs)
        records.append({"image": path.name, "raw_max_abs": frame_max, "pt_boxes": len(pt_boxes), "onnx_boxes": len(onnx_boxes), "matched": len(pairs)})
    result = {
        "pt_model": str(args.pt_model.resolve()), "onnx_model": str(args.onnx_model.resolve()),
        "onnx_input": {"name": inp.name, "shape": inp.shape}, "onnx_output": {"name": out.name, "shape": out.shape},
        "images": len(image_paths), "class_id": 8, "conf": 0.05,
        "raw": {"max_abs_difference": max_abs, "mean_abs_at_max_frame": max_mean, "max_frame": max_frame},
        "decoded": {"matched": len(ious), "mean_iou": float(np.mean(ious)) if ious else None,
                    "mean_coordinate_abs_error_px": float(np.mean(coord_errors)) if coord_errors else None,
                    "mean_confidence_abs_error": float(np.mean(conf_errors)) if conf_errors else None,
                    "unmatched_pytorch": unmatched_pt, "unmatched_onnx": unmatched_onnx},
        "frames": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
