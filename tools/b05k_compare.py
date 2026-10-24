"""Compare PyTorch and ONNX Runtime on identical source-video tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.b05k_common import box_iou, decode_yolo11, prepare_input  # noqa: E402
from tools.b05k_runtime import ModelRuntime, resolve_model_path  # noqa: E402


def _numpy(output: Any) -> np.ndarray:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if hasattr(output, "detach"):
        output = output.detach().float().cpu().numpy()
    return np.asarray(output, dtype=np.float32)


def _match(left: list[dict], right: list[dict], threshold: float = 0.5) -> list[tuple[dict, dict, float]]:
    remaining = set(range(len(right)))
    pairs = []
    for a in sorted(left, key=lambda item: -item["confidence"]):
        choices = [(box_iou(a["bbox"], right[index]["bbox"]), index) for index in remaining]
        if not choices:
            continue
        overlap, index = max(choices)
        if overlap >= threshold:
            remaining.remove(index)
            pairs.append((a, right[index], overlap))
    return pairs


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--pt-model", type=Path)
    parser.add_argument("--onnx-model", type=Path)
    parser.add_argument("--model-manifest", type=Path, help="manifest containing both source model hashes")
    parser.add_argument("--preprocess", choices=("letterbox", "resize"), default="letterbox")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--failure-tolerance", type=float, default=1.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--input-height", type=int, default=192)
    parser.add_argument("--roi-top", type=float, default=0.60)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.model_manifest:
        pt_path = resolve_model_path(backend="pytorch", manifest_path=args.model_manifest)
        onnx_path = resolve_model_path(backend="onnx", manifest_path=args.model_manifest)
    elif args.pt_model and args.onnx_model:
        pt_path, onnx_path = args.pt_model.resolve(), args.onnx_model.resolve()
    else:
        parser.error("provide both --pt-model and --onnx-model, or use --model-manifest")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {args.output} (use --overwrite)")
    if not args.video.is_file():
        raise FileNotFoundError(args.video)

    pt = ModelRuntime("pytorch", pt_path, device=args.device)
    onnx = ModelRuntime("onnx", onnx_path, device="auto")
    pt.validate_canvas(args.input_width, args.input_height)
    onnx.validate_canvas(args.input_width, args.input_height)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    max_abs = 0.0
    max_mean = 0.0
    max_diff_frame = None
    frame_records = []
    box_ious = []
    box_abs_errors = []
    score_abs_errors = []
    raw_times = {"pytorch": [], "onnx": []}
    processed = 0
    source_index = 0
    input_hashes = set()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index = source_index
        source_index += 1
        if frame_index % args.stride:
            continue
        tensor, transform = prepare_input(
            frame, input_width=args.input_width, input_height=args.input_height,
            roi_top=args.roi_top, preprocess=args.preprocess,
        )
        input_hash = hashlib.sha256(tensor.tobytes()).hexdigest()
        input_hashes.add(input_hash)
        pt_output, pt_ms = pt.run(tensor)
        onnx_output, onnx_ms = onnx.run(tensor)
        raw_pt, raw_onnx = _numpy(pt_output), _numpy(onnx_output)
        if raw_pt.shape != raw_onnx.shape:
            raise RuntimeError(f"raw output shape mismatch: {raw_pt.shape} vs {raw_onnx.shape}")
        difference = np.abs(raw_pt - raw_onnx)
        frame_max, frame_mean = float(difference.max(initial=0.0)), float(difference.mean())
        if frame_max > max_abs:
            max_abs, max_mean, max_diff_frame = frame_max, frame_mean, frame_index
        raw_times["pytorch"].append(pt_ms)
        raw_times["onnx"].append(onnx_ms)

        pt_boxes = decode_yolo11(raw_pt, conf=args.conf, iou=args.iou)
        onnx_boxes = decode_yolo11(raw_onnx, conf=args.conf, iou=args.iou)
        for item in pt_boxes + onnx_boxes:
            item["bbox"] = transform.network_to_frame(item["bbox"])
        pairs = _match(pt_boxes, onnx_boxes, threshold=args.match_iou)
        for a, b, overlap in pairs:
            box_ious.append(overlap)
            box_abs_errors.append(float(np.mean(np.abs(np.asarray(a["bbox"]) - np.asarray(b["bbox"])))))
            score_abs_errors.append(abs(float(a["confidence"]) - float(b["confidence"])))
        frame_records.append({
            "frame": frame_index,
            "input_sha256": input_hash,
            "raw_max_abs": frame_max,
            "raw_mean_abs": frame_mean,
            "pytorch_detections": len(pt_boxes),
            "onnx_detections": len(onnx_boxes),
            "matched_boxes": len(pairs),
        })

        if frame_max >= args.failure_tolerance and args.diagnostics_dir:
            diagnostics = args.diagnostics_dir.resolve()
            diagnostics.mkdir(parents=True, exist_ok=True)
            stem = f"frame_{frame_index:06d}_rawdiff_{frame_max:.4f}"
            np.savez_compressed(
                diagnostics / f"{stem}.npz",
                input_tensor=tensor,
                pytorch_output=raw_pt,
                onnx_output=raw_onnx,
                pytorch_boxes=np.asarray([item["bbox"] for item in pt_boxes], dtype=np.float32),
                onnx_boxes=np.asarray([item["bbox"] for item in onnx_boxes], dtype=np.float32),
            )
            cv2.imwrite(str(diagnostics / f"{stem}.jpg"), frame)
        processed += 1
        if args.limit and processed >= args.limit:
            break

    capture.release()
    summary = {
        "video": str(args.video.resolve()),
        "video_name": args.video.name,
        "source_size": [width, height],
        "source_fps": fps,
        "metadata_frame_count_hint": max_frames,
        "processed_frames": len(frame_records),
        "preprocess": args.preprocess,
        "input_size": [args.input_height, args.input_width],
        "same_tensor_instance_sent_to_both_backends": True,
        "unique_input_hashes": len(input_hashes),
        "pytorch": {"model": str(pt_path), "sha256": _sha(pt_path), "device": pt.device, "providers": pt.providers,
                    "mean_inference_ms": float(np.mean(raw_times["pytorch"])) if raw_times["pytorch"] else None},
        "onnx": {"model": str(onnx_path), "sha256": _sha(onnx_path), "device": onnx.device, "providers": onnx.providers,
                 "mean_inference_ms": float(np.mean(raw_times["onnx"])) if raw_times["onnx"] else None},
        "raw_output": {"max_abs_difference": max_abs, "mean_abs_at_max_frame": max_mean, "max_difference_frame": max_diff_frame},
        "decoded_boxes": {
            "matched_pairs_iou_ge": args.match_iou,
            "matched_pairs": len(box_ious),
            "mean_iou": float(np.mean(box_ious)) if box_ious else None,
            "mean_coordinate_abs_error_px": float(np.mean(box_abs_errors)) if box_abs_errors else None,
            "mean_confidence_abs_error": float(np.mean(score_abs_errors)) if score_abs_errors else None,
            "unmatched_pytorch": sum(row["pytorch_detections"] for row in frame_records) - len(box_ious),
            "unmatched_onnx": sum(row["onnx_detections"] for row in frame_records) - len(box_ious),
        },
        "failure_tolerance": args.failure_tolerance,
        "diagnostics_saved_for_frames_over_tolerance": bool(args.diagnostics_dir),
        "frames": frame_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "frames"}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
