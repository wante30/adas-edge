"""FRONT-only traffic-light specialist: annotated video on the board.

Loads the hybrid-quantized specialist RKNN, runs the deployment path on a road
video (top-60% ROI crop -> letterbox 512x192 -> NPU -> decode -> map back to
full-frame coordinates) and writes an annotated MP4 plus a per-frame summary.
Python 3.8 compatible; expects cv2 + rknnlite on the board.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_yolo_output import decode_outputs  # noqa: E402


def letterbox(image, size):
    width, height = size
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = round(w * scale), round(h * scale)
    pad_x, pad_y = (width - nw) // 2, (height - nh) // 2
    resized = cv2.resize(image, (nw, nh))
    canvas = np.full((height, width, 3), 114, np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--roi-top", type=float, default=0.60)
    args = parser.parse_args()

    from rknnlite.api import RKNNLite

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rknn = RKNNLite()
    assert rknn.load_rknn(args.model) == 0, "load_rknn failed"
    assert rknn.init_runtime() == 0, "init_runtime failed"  # core_mask is RK3588-only; RK3566 has one NPU core

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), "cannot open video"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_h = int(round(height * args.roi_top))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(
        str(out_dir / "tl_specialist_annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )

    rows = []
    total = 0
    index = 0
    while index < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[:roi_h, :, :]
        canvas, scale, pad_x, pad_y = letterbox(crop, (512, 192))
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        outputs = rknn.inference(inputs=[np.expand_dims(rgb, 0)])
        detections = decode_outputs(outputs, args.conf, args.iou)
        drawn = 0
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            fx1 = int(round((x1 - pad_x) / scale))
            fy1 = int(round((y1 - pad_y) / scale))
            fx2 = int(round((x2 - pad_x) / scale))
            fy2 = int(round((y2 - pad_y) / scale))
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
            cv2.putText(frame, "TL %.2f" % det["confidence"], (fx1, max(12, fy1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            drawn += 1
        writer.write(frame)
        rows.append({"frame": index, "detections": drawn,
                     "scores": [round(float(d["confidence"]), 4) for d in detections]})
        total += drawn
        index += 1

    cap.release()
    writer.release()
    rknn.release()

    with (out_dir / "frame_stats.csv").open("w", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=["frame", "detections", "scores"])
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    summary = {
        "model": args.model,
        "video": args.video,
        "frames": index,
        "conf": args.conf,
        "roi_top": args.roi_top,
        "canvas": "512x192",
        "total_detections": total,
        "avg_per_frame": total / max(index, 1),
        "frames_with_detection": sum(1 for r in rows if r["detections"] > 0),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
