"""Inspect one RKNN model output against a saved Qt MFR1 frame snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import struct

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--frame-snapshot", required=True)
    parser.add_argument("--quadrant", type=int, choices=range(4), required=True)
    parser.add_argument("--root", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.root:
        sys.path.insert(0, str(Path(args.root).resolve()))
    from competition_ai.detector import CompetitionDetector

    snapshot = Path(args.frame_snapshot).read_bytes()
    model_path = Path(args.model).resolve()
    frame_sha256 = hashlib.sha256(snapshot).hexdigest()
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    header = struct.Struct("<4sQQII")
    frame_bytes = 1280 * 720 * 2
    if len(snapshot) < header.size:
        raise ValueError("frame snapshot is shorter than the MFR1 header")
    magic, frame_id, timestamp_ns, stride_bytes, payload_bytes = header.unpack_from(snapshot)
    if (magic, stride_bytes, payload_bytes, len(snapshot)) != (
        b"MFR1", 1280 * 2, frame_bytes, header.size + frame_bytes
    ):
        raise ValueError("MFR1 header or frame payload validation failed")

    from deployment.rknn.decode_raw_heads import (
        decode_raw_heads_thresholded,
        looks_like_raw_head_triplets,
    )

    detector = CompetitionDetector(
        args.model,
        input_size=(512, 288),
        class_profile="bdd10",
        use_rga_preprocess=True,
    )
    try:
        packed = np.frombuffer(snapshot[header.size:], dtype=np.uint8).reshape(720, 1280, 2)
        row, column = divmod(args.quadrant, 2)
        roi = packed[row * 360:(row + 1) * 360, column * 640:(column + 1) * 640]
        frame = cv2.cvtColor(roi, cv2.COLOR_BGR5652BGR)
        image, scale, pad_x, pad_y = detector.preprocess(frame)
        outputs = detector._runner.inference(
            inputs=[image[None, ...]], data_format="nhwc"
        )
        detections = decode_raw_heads_thresholded(
            outputs,
            input_width=int(detector.input_size[0]),
            input_height=int(detector.input_size[1]),
            confidence=detector.conf_threshold,
            iou=detector.nms_threshold,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )
        rows = []
        for index, output in enumerate(outputs):
            value = np.asarray(output, dtype=np.float32)
            item = {
                "index": index,
                "shape": list(value.shape),
                "min": float(np.nanmin(value)),
                "max": float(np.nanmax(value)),
                "mean": float(np.nanmean(value)),
            }
            if index % 3 == 1:
                if value.ndim == 4 and value.shape[1] <= 20:
                    class_scores = value[0]
                elif value.ndim == 4 and value.shape[-1] <= 20:
                    class_scores = value[0].transpose(2, 0, 1)
                else:
                    class_scores = None
                if class_scores is not None:
                    maxima = class_scores.reshape(class_scores.shape[0], -1).max(axis=0)
                    item["class_max"] = {
                        str(threshold): int(np.count_nonzero(maxima >= threshold))
                        for threshold in (0.05, 0.10, 0.25, 0.50, 0.75)
                    }
                    item["anchor_max_p99"] = float(np.percentile(maxima, 99))
            rows.append(item)
        print(json.dumps({
            "frame_id": int(frame_id),
            "timestamp_ns": int(timestamp_ns),
            "frame_snapshot_sha256": frame_sha256,
            "model_path": str(model_path),
            "model_sha256": model_sha256,
            "quadrant": args.quadrant,
            "source_shape": list(frame.shape),
            "model_input_shape": list(image.shape),
            "preprocess_backend": detector.last_preprocess_backend,
            "scale": scale,
            "pad": [pad_x, pad_y],
            "output_count": len(outputs),
            "raw_head_triplets": looks_like_raw_head_triplets(outputs),
            "confidence_threshold": detector.conf_threshold,
            "nms_iou_threshold": detector.nms_threshold,
            "detection_count": len(detections),
            "detections": [
                {
                    "class_id": int(row[5]),
                    "class_name": detector.class_names[int(row[5])],
                    "confidence": float(row[4]),
                    "bbox": [float(value) for value in row[:4]],
                }
                for row in detections
            ],
            "outputs": rows,
        }, indent=2), flush=True)
    finally:
        detector.close()


if __name__ == "__main__":
    main()
