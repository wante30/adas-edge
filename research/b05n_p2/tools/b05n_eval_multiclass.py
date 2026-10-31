"""B-05N ten-class holdout evaluation via Ultralytics validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_model(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=parse_model, action="append", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torchvision

    _orig_nms = torchvision.ops.nms

    def nms_cpu_fallback(boxes, scores, iou_threshold):
        if boxes.is_cuda:
            return _orig_nms(boxes.cpu(), scores.cpu(), iou_threshold).to(boxes.device)
        return _orig_nms(boxes, scores, iou_threshold)

    if torchvision.__version__.endswith("+cpu"):
        torchvision.ops.nms = nms_cpu_fallback
    from ultralytics import YOLO

    results = {"data": str(args.data.resolve()), "split": args.split, "imgsz": args.imgsz, "models": {}}
    for label, path in args.model:
        model = YOLO(str(path))
        metrics = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz, rect=True,
                            batch=args.batch, device=args.device, conf=0.001, iou=0.5,
                            plots=False, save_json=False, verbose=False)
        names = {int(key): value for key, value in metrics.names.items()}
        per_class = {}
        for index, name in names.items():
            per_class[name] = {
                "ap50": float(metrics.box.ap50[index]) if hasattr(metrics.box, "ap50") else None,
                "ap50_95": float(metrics.box.maps[index]),
                "precision": float(metrics.box.p[index]),
                "recall": float(metrics.box.r[index]),
                "f1": float(2 * metrics.box.p[index] * metrics.box.r[index] /
                            max(1e-9, metrics.box.p[index] + metrics.box.r[index])),
            }
        results["models"][label] = {
            "path": str(path.resolve()),
            "classes": names,
            "overall": {
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
                "precision": float(np.mean(metrics.box.p)),
                "recall": float(np.mean(metrics.box.r)),
            },
            "per_class": per_class,
            "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
        }
        print(f"DONE {label} mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
