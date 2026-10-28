"""Compare frozen main/specialist models on an actual competition video.

PC uses the static split ONNX graphs from the RKNN build, with identical
RGB letterboxing and coordinate restoration. No training or fake labels.
Raw predictions are cached so changes to postprocessing can be compared
without changing model/input. This is not an RK3566 timing result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from deployment.rknn.decode_yolo_output import decode_outputs
from tools.profile_rknn_single_stream import letterbox_bgr
from competition_ai.traffic_light import classify_traffic_light


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class StaticOnnx:
    def __init__(self, path, shape, tl_id, source):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        meta = self.session.get_inputs()[0]
        assert meta.shape == [1, 3, shape[1], shape[0]], meta.shape
        self.name, self.shape, self.tl_id, self.source = meta.name, shape, tl_id, source

    def detect(self, frame):
        started = time.perf_counter()
        rgb, scale, px, py = letterbox_bgr(frame, self.shape)
        x = np.transpose(rgb.astype(np.float32) / 255, (2, 0, 1))[None]
        outputs = self.session.run(None, {self.name: x})
        detections = decode_outputs(outputs, 0.001, 0.7)
        result = []
        h, w = frame.shape[:2]
        for item in detections:
            if item["class_id"] != self.tl_id:
                continue
            x1, y1, x2, y2 = item["bbox"]
            result.append({"class_name": "traffic light", "confidence": item["confidence"],
                           "bbox": [max(0., min(w, (x1-px)/scale)), max(0., min(h, (y1-py)/scale)),
                                    max(0., min(w, (x2-px)/scale)), max(0., min(h, (y2-py)/scale))],
                           "source": self.source})
        return result, (time.perf_counter() - started) * 1000


def annotate(frame, items, title):
    view = cv2.resize(frame, (960, 540))
    sx, sy = 960 / frame.shape[1], 540 / frame.shape[0]
    for item in items:
        if item["confidence"] < 0.05:
            continue
        x1, y1, x2, y2 = item["bbox"]
        a, b = (round(x1*sx), round(y1*sy)), (round(x2*sx), round(y2*sy))
        color = (0, 210, 255) if item["confidence"] >= .25 else (210, 130, 30)
        cv2.rectangle(view, a, b, color, 1)
        cv2.putText(view, f'{item["confidence"]:.2f}', (a[0], max(15, a[1]-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1)
    cv2.rectangle(view, (0, 505), (960, 540), (15, 15, 15), -1)
    cv2.putText(view, title, (12, 529), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1)
    return view


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--main-model", type=Path, required=True)
    p.add_argument("--specialist-model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--specialist-roi", choices=("top60", "focus"), default="top60")
    p.add_argument("--main-cache", type=Path)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_predictions.jsonl"
    if raw_path.exists():
        raise SystemExit("Use a fresh output directory; preserving existing evidence")
    main_model = StaticOnnx(args.main_model, (512, 288), 9, "MAIN_ONNX_PC")
    specialist = StaticOnnx(args.specialist_model, (512, 192), 0, "TRAFFIC_LIGHT_SPECIALIST_PC")
    main_cache = ({r['frame']: r for r in map(json.loads, args.main_cache.open(encoding='utf-8'))}
                  if args.main_cache else None)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot read {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(args.frames or total, total)
    video = cv2.VideoWriter(str(args.output_dir / "main_vs_specialist.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (1920, 540))
    snapshot_frames = {0, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1180, 1200, 1238, 1250, 1270, 1348}
    rows = []
    started = time.perf_counter()
    with raw_path.open("w", encoding="utf-8") as raw:
        for fid in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if fid % args.stride:
                continue
            if main_cache is None:
                m, main_ms = main_model.detect(frame)
            else:
                m, main_ms = main_cache[fid]['main'], main_cache[fid]['main_ms']
            h,w=frame.shape[:2]
            x0,y0,x1,y1=(0,0,w,round(h*.6)) if args.specialist_roi=='top60' else (round(w*.2),round(h*.25),round(w*.8),round(h*.65))
            s, signal_ms = specialist.detect(frame[y0:y1,x0:x1])
            for item in s:
                bx1,by1,bx2,by2=item['bbox']; item['bbox']=[bx1+x0,by1+y0,bx2+x0,by2+y0]
            for item in m+s:
                item["raw_color"] = str(classify_traffic_light(frame, item, min_confidence=0.05))
            raw.write(json.dumps({"frame": fid, "source_s": fid/fps, "main": m, "specialist": s,
                                  "main_ms": main_ms, "specialist_ms": signal_ms}) + "\n")
            rows.append({"frame": fid, "main_ms": round(main_ms, 3), "specialist_ms": round(signal_ms, 3),
                         **{f"{name}_count_{threshold}": sum(i["confidence"] >= threshold for i in items)
                            for name, items in [("main", m), ("specialist", s)] for threshold in [.001, .05, .15, .25]}})
            canvas = np.hstack([annotate(frame, m, f"FRAME {fid} | MAIN YOLO11n 512x288"),
                                annotate(frame, s, f"FRAME {fid} | SPECIALIST 512x192 top60%")])
            video.write(canvas)
            if fid in snapshot_frames:
                cv2.imwrite(str(args.output_dir/f"compare_{fid:04d}.jpg"), canvas)
            if fid % 150 == 0:
                print(f"{fid}/{total} frames, main {main_ms:.1f} ms, specialist {signal_ms:.1f} ms", flush=True)
    video.release()
    cap.release()
    with (args.output_dir / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"evidence": "PC_STATIC_ONNX_CPU_NOT_RK3566", "video": str(args.video.resolve()),
               "video_sha256": sha256(args.video), "main_model": str(args.main_model.resolve()),
               "main_sha256": sha256(args.main_model), "specialist_model": str(args.specialist_model.resolve()),
               "specialist_sha256": sha256(args.specialist_model), "frames": len(rows), "source_fps": fps,
               "stride": args.stride, "wall_seconds": time.perf_counter()-started,
               "specialist_roi": args.specialist_roi,
               "main_predictions_reused_from": str(args.main_cache) if args.main_cache else None,
               "latency": {key: {"mean": float(np.mean([r[key] for r in rows])),
                                  "p95": float(np.percentile([r[key] for r in rows], 95))}
                           for key in ["main_ms", "specialist_ms"]}}
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
