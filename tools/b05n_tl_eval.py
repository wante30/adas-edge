"""B-05N full-frame traffic-light evaluation on BDD50.

The ten-class model sees the entire 1280x720 frame resized to 512x288.  We
decode only class 8 (traffic light), apply NMS, map boxes back to original
pixels, and report both the B-05L 150-GT ROI metric and the 154-GT full-frame
metric.  This is deliberately separate from the single-class specialist input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.b05k_common import _nms, evaluate_predictions  # noqa: E402
from tools.b05l_eval import load_gt  # noqa: E402

DEFAULT_IMAGES = Path(r"D:\Datasets\adas_edge_training\b03c_rknn_validation_20261021\board_holdout50")
DEFAULT_GT = Path(r"D:\Datasets\adas_edge_training\b03d_gt_hf_20261021")


def prepare_full(frame: np.ndarray, width: int, height: int) -> tuple[np.ndarray, dict[str, float]]:
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, {"scale_x": width / frame.shape[1], "scale_y": height / frame.shape[0]}


def decode_class(raw: Any, class_id: int, conf: float, iou: float, max_det: int = 300) -> list[dict[str, Any]]:
    if isinstance(raw, (tuple, list)):
        raw = raw[0]
    if hasattr(raw, "detach"):
        raw = raw.detach().float().cpu().numpy()
    array = np.asarray(raw, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[0] < 5:
        raise ValueError(f"expected [4+nc,N], got {array.shape}")
    scores = array[4 + class_id]
    if scores.min(initial=0.0) < -1e-4 or scores.max(initial=0.0) > 1.0001:
        raise ValueError("class scores are outside [0,1]")
    ids = np.flatnonzero(scores >= conf)
    if not ids.size:
        return []
    cx, cy, bw, bh = array[:4, ids]
    boxes = np.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), axis=1)
    valid = (bw > 0) & (bh > 0)
    boxes, scores = boxes[valid], scores[ids][valid]
    if not len(boxes):
        return []
    keep = _nms(boxes, scores, iou, max_det)
    return [{"bbox": boxes[i].astype(float).tolist(), "confidence": float(scores[i])} for i in keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--input-height", type=int, default=288)
    parser.add_argument("--roi-top", type=float, default=0.60)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--ap-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    from ultralytics import YOLO

    samples, gt_roi, counts = load_gt(args.images, args.gt_dir, args.roi_top)
    # The loader only keys ROI targets; full-frame GT is recovered by re-reading JSON.
    gt_full: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(args.gt_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        name = str(data["name"])
        targets = []
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                if obj.get("category") != "traffic light":
                    continue
                b = obj["box2d"]
                x1, y1, x2, y2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
                if x2 > x1 and y2 > y1:
                    targets.append({"bbox": [x1, y1, x2, y2], "width_px": x2 - x1})
        gt_full[name] = targets

    model = YOLO(str(args.model)).model.to(args.device).eval()
    import torch

    times = []
    predictions = {}
    for sample in samples:
        frame = cv2.imread(str(sample["path"]))
        tensor, transform = prepare_full(frame, args.input_width, args.input_height)
        input_tensor = torch.from_numpy(tensor).to(args.device)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            raw = model(input_tensor)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1000.0)
        decoded = decode_class(raw, 8, args.ap_floor, args.nms_iou)
        for item in decoded:
            x1, y1, x2, y2 = item["bbox"]
            item["bbox"] = [x1 / transform["scale_x"], y1 / transform["scale_y"],
                            x2 / transform["scale_x"], y2 / transform["scale_y"]]
        predictions[sample["name"]] = decoded

    roi_height = int(round(720 * args.roi_top))
    predictions_roi = {
        name: [item for item in items if (item["bbox"][1] + item["bbox"][3]) / 2 < roi_height]
        for name, items in predictions.items()
    }
    metrics_roi = evaluate_predictions(gt_roi, predictions_roi, conf=args.conf, iou=args.match_iou, ap_floor=args.ap_floor)
    metrics_full = evaluate_predictions(gt_full, predictions, conf=args.conf, iou=args.match_iou, ap_floor=args.ap_floor)
    result = {
        "model": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "input": {"width": args.input_width, "height": args.input_height, "full_frame": True, "rgb": True, "scale": "1/255"},
        "dataset": {**counts, "images": len(samples), "roi_height": roi_height},
        "thresholds": {"conf": args.conf, "ap_floor": args.ap_floor, "nms_iou": args.nms_iou, "match_iou": args.match_iou},
        "metrics_roi": metrics_roi,
        "metrics_full": metrics_full,
        "detections": {name: len(items) for name, items in predictions.items()},
        "runtime": {"mean_ms": statistics.fmean(times), "p50_ms": float(np.percentile(times, 50)),
                    "p95_ms": float(np.percentile(times, 95))},
        "predictions_by_image": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("dataset", "thresholds", "metrics_roi", "metrics_full", "runtime")}, indent=2))


if __name__ == "__main__":
    main()
