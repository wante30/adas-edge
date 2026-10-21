"""Collect Ultralytics ONNX predictions in the BDD evaluator's dump format."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bdd-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()

    if args.width < 32 or args.height < 32 or args.width % 32 or args.height % 32:
        parser.error(f"canvas must be >=32 and divisible by 32, got {args.width}x{args.height}")
    if args.limit < 1 or args.max_det < 1:
        parser.error("limit and max-det must be positive")
    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    if args.out.exists():
        parser.error(f"refusing to overwrite existing output: {args.out}")

    import onnx
    from ultralytics import YOLO

    graph = onnx.load(str(args.model), load_external_data=False)
    if len(graph.graph.input) != 1:
        parser.error(f"expected one ONNX input, found {len(graph.graph.input)}")
    input_dims = [dim.dim_value or dim.dim_param for dim in graph.graph.input[0].type.tensor_type.shape.dim]
    expected_shape = [1, 3, args.height, args.width]
    if input_dims != expected_shape:
        parser.error(f"ONNX input shape {input_dims} does not match requested NCHW shape {expected_shape}")

    image_root = args.bdd_root / "images" / "100k" / "val"
    image_paths = sorted(image_root.glob("*.jpg"))[: args.limit]
    if not image_paths:
        raise SystemExit(f"no BDD images under {image_root}")

    model = YOLO(str(args.model), task="detect")
    predict_args = {
        "imgsz": (args.height, args.width),
        "conf": args.score_floor,
        "max_det": args.max_det,
        "iou": args.iou,
        "device": args.device,
        "rect": False,
        "verbose": False,
    }
    model.predict(source=str(image_paths[0]), **predict_args)

    per_image = []
    started = time.perf_counter()
    for path in image_paths:
        result = model.predict(source=str(path), **predict_args)[0]
        detections = [
            [int(class_id), float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score)]
            for box, class_id, score in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            )
        ]
        per_image.append({"image": path.name, "detections": detections})
    elapsed = time.perf_counter() - started

    payload = {
        "model": str(args.model),
        "width": args.width,
        "height": args.height,
        "score_floor": args.score_floor,
        "max_det": args.max_det,
        "iou": args.iou,
        "observed_input_shapes": [expected_shape],
        "model_names": {str(index): name for index, name in model.names.items()},
        "per_image": per_image,
        "runtime": {
            "device": args.device,
            "wall_ms_per_image": elapsed * 1000 / len(image_paths),
            "wall_total_s": elapsed,
            "timing_scope": "PC ONNX Runtime, one image per forward, includes preprocessing and postprocessing",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    print(json.dumps({"images": len(image_paths), "input_shape": expected_shape, "out": str(args.out), **payload["runtime"]}, indent=2))


if __name__ == "__main__":
    main()
