"""Run frozen ONNX exports at RK3566 source-time cadences on a complete video.

This is an offline PC FP32 configuration simulation, not RKNN INT8 execution
or a simulation of board compute latency. All displayed boxes are detections;
between inference updates their source coordinates are held without tracking.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
from competition_ai.detector import COCO_NAMES, COMPETITION_CLASSES
from competition_ai.traffic_light import classify_traffic_light
from deployment.rknn.decode_yolo_output import decode_outputs
from tools.profile_rknn_single_stream import letterbox_bgr
from tools.adas_hmi_sim.backend.traffic_light_runtime import TrafficLightEstimate, TrafficLightTemporalFilter
from tools.adas_hmi_sim.backend.traffic_light_specialist import SignalCandidateCache, next_signal_roi, signal_refresh_due


class FrozenOnnx:
    def __init__(self, path: Path, shape: tuple[int, int], specialist: bool = False):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        meta = self.session.get_inputs()[0]
        if meta.shape != [1, 3, shape[1], shape[0]]:
            raise ValueError(f"Unexpected static model input: {meta.shape}")
        self.name, self.shape, self.specialist = meta.name, shape, specialist

    def detect(self, frame: np.ndarray) -> tuple[list[dict], float]:
        start = time.perf_counter()
        rgb, scale, px, py = letterbox_bgr(frame, self.shape)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.
        output = self.session.run(None, {self.name: tensor})
        decoded = decode_outputs(output, .05 if self.specialist else .25, .70 if self.specialist else .45)
        height, width = frame.shape[:2]
        items = []
        for item in decoded:
            cid = item["class_id"]
            if self.specialist and cid != 0:
                raise ValueError(f"Unexpected specialist class: {cid}")
            name = "traffic light" if self.specialist else COCO_NAMES[cid]
            if name not in COMPETITION_CLASSES:
                continue
            x1, y1, x2, y2 = item["bbox"]
            box = [float(np.clip((x1-px)/scale, 0, width)), float(np.clip((y1-py)/scale, 0, height)),
                   float(np.clip((x2-px)/scale, 0, width)), float(np.clip((y2-py)/scale, 0, height))]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            items.append({"class_id": cid, "class_name": name, "confidence": item["confidence"], "bbox": box,
                          "source": "TRAFFIC_LIGHT_SPECIALIST_PC" if self.specialist else "MAIN_ONNX_PC"})
        return items, (time.perf_counter()-start)*1000


COLORS = {"car": (255, 190, 30), "truck": (255, 100, 80), "bus": (255, 100, 80),
          "person": (50, 180, 255), "motorcycle": (200, 90, 255), "bicycle": (200, 90, 255),
          "stop sign": (40, 40, 255)}
SIGNAL_COLORS = {"RED": (60, 60, 255), "GREEN": (60, 240, 90), "YELLOW": (40, 225, 255), "UNKNOWN": (180, 180, 180)}


def render(frame, main_items, signals, estimate, fid, fps, main_age, spec_age, main_hz, spec_hz, out_size):
    view = cv2.resize(frame, out_size)
    sx, sy = out_size[0]/frame.shape[1], out_size[1]/frame.shape[0]
    font_scale = out_size[0]/1920
    for item in main_items + signals:
        x1, y1, x2, y2 = item["bbox"]
        a = (round(x1*sx), round(y1*sy)); b = (round(x2*sx), round(y2*sy))
        name = item["class_name"]
        if name == "traffic light":
            if item["source"] == "TRAFFIC_LIGHT_SPECIALIST_PC":
                state = item["lamp_evidence_state"] if item["lamp_evidence_valid"] else "UNKNOWN"
                label = f'TL-S {state} {item["confidence"]:.2f}'
            else:
                state = item.get("display_color", "UNKNOWN")
                label = f'TL-M {state} {item["confidence"]:.2f}'
            color = SIGNAL_COLORS[state]
        else:
            color = COLORS.get(name, (255, 255, 255))
            label = f'{name} {item["confidence"]:.2f}'
        cv2.rectangle(view, a, b, color, 2)
        scale = .55*font_scale
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = max(th+4, a[1]-4)
        lx = min(max(0, a[0]), max(0, out_size[0]-tw-4))
        cv2.rectangle(view, (lx, ty-th-3), (lx+tw+4, ty+3), (20, 25, 30), -1)
        cv2.putText(view, label, (lx+2, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    # The footer reports source-time cache age, not a fabricated NPU latency.
    panel_h = round(90*font_scale)
    y0 = out_size[1]-panel_h
    cv2.rectangle(view, (0, y0), (out_size[0], out_size[1]), (18, 24, 30), -1)
    lines = [f'PC FP32 CONFIG SIM | MAIN 512x288 @ {main_hz:g} Hz | TL 512x192 @ {spec_hz:g} Hz',
             f'TIME {fid/fps:6.2f}s | FRAME {fid} | BOX SOURCE AGE main {main_age:.0f}ms / TL {spec_age:.0f}ms',
             f'OBSERVED SIGNAL {estimate.state} | VERTICAL SIGNAL {estimate.pedestrian_state} | EGO ASSOCIATION UNVERIFIED']
    for i, line in enumerate(lines):
        cv2.putText(view, line, (round(14*font_scale), y0+round((23+28*i)*font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, .60*font_scale, (235, 240, 245), 1, cv2.LINE_AA)
    return view


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def statistics(values):
    return {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)), "max": float(max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--main-model", type=Path, required=True)
    parser.add_argument("--specialist-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--main-fps", type=float, default=10.)
    parser.add_argument("--specialist-fps", type=float, default=3.)
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes the entire source once")
    parser.add_argument("--output-width", type=int, default=1920)
    args = parser.parse_args()
    if args.main_fps <= 0 or args.specialist_fps <= 0:
        parser.error("Cadences must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir/name).exists() for name in ("annotated_mp4v.mp4", "predictions.jsonl", "summary.json")):
        raise SystemExit("Use a fresh output directory; existing evidence is preserved")
    detector = FrozenOnnx(args.main_model, (512, 288))
    specialist = FrozenOnnx(args.specialist_model, (512, 192), True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    if fps <= 0 or total <= 0 or min(width, height) <= 0:
        raise RuntimeError("Invalid source video metadata")
    limit = min(total, args.max_frames or total)
    out_size = (args.output_width, round(height*args.output_width/width/2)*2)
    writer = cv2.VideoWriter(str(args.output_dir/"annotated_mp4v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    if not writer.isOpened():
        raise RuntimeError("Cannot create annotated video")
    cache = SignalCandidateCache()
    temporal = TrafficLightTemporalFilter(ego_x_max=.95, pedestrian_hold_frames=0)
    main_items, last_main, last_spec = [], None, None
    next_roi, roi_counts = "top60", Counter()
    main_ms, spec_ms, color_counts, class_counts = [], [], Counter(), Counter()
    estimate = TrafficLightEstimate()
    snapshots = {round(v*(limit-1)) for v in (0, .125, .25, .375, .5, .625, .75, .875, 1)}
    main_max_age, spec_max_age = 0., 0.
    started = time.perf_counter()
    processed = 0
    try:
        with (args.output_dir/"predictions.jsonl").open("w", encoding="utf-8") as raw:
            for fid in range(limit):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Early decode failure at {fid}/{limit}")
                t = fid/fps
                main_due = signal_refresh_due(t, last_main, args.main_fps)
                spec_due = signal_refresh_due(t, last_spec, args.specialist_fps)
                if main_due:
                    main_items, elapsed = detector.detect(frame)
                    main_ms.append(elapsed); last_main = t
                    for item in main_items:
                        class_counts[item["class_name"]] += 1
                        if item["class_name"] == "traffic light":
                            item["display_color"] = classify_traffic_light(frame, item, require_lamp_evidence=True).value
                if spec_due:
                    used_roi = next_roi
                    x0, y0, x1, y1 = ((0, 0, width, round(height*.6)) if used_roi == "top60" else
                                      (round(width*.2), round(height*.25), round(width*.8), round(height*.65)))
                    candidates, elapsed = specialist.detect(frame[y0:y1, x0:x1])
                    spec_ms.append(elapsed); last_spec = t; roi_counts[used_roi] += 1
                    for item in candidates:
                        bx1, by1, bx2, by2 = item["bbox"]
                        item["bbox"] = [bx1+x0, by1+y0, bx2+x0, by2+y0]
                    cache.store(frame, candidates, frame_id=fid, timestamp_ms=t*1000)
                    next_roi = next_signal_roi(frame, candidates, used_roi)
                signals = cache.current(t*1000)
                if main_due:
                    estimate = temporal.update(frame, main_items + signals)
                ma, sa = (t-last_main)*1000, (t-last_spec)*1000
                main_max_age, spec_max_age = max(main_max_age, ma), max(spec_max_age, sa)
                color_counts[estimate.state] += 1
                raw.write(json.dumps({"frame": fid, "source_s": t, "main_fresh": main_due, "specialist_fresh": spec_due,
                                      "main_source_frame": round(last_main*fps), "specialist_source_frame": round(last_spec*fps),
                                      "main_age_ms": round(ma, 3), "specialist_age_ms": round(sa, 3),
                                      "main": main_items, "specialist": signals, "signal_roi": used_roi,
                                      "signal": estimate.as_metadata()})+"\n")
                canvas = render(frame, main_items, signals, estimate, fid, fps, ma, sa,
                                args.main_fps, args.specialist_fps, out_size)
                writer.write(canvas); processed += 1
                if fid in snapshots:
                    cv2.imwrite(str(args.output_dir/f"frame_{fid:05d}.jpg"), canvas)
                if fid % 300 == 0:
                    print(f"{fid}/{limit} source frames | main updates {len(main_ms)}, TL updates {len(spec_ms)} | elapsed {time.perf_counter()-started:.1f}s", flush=True)
    finally:
        writer.release(); cap.release()
    summary = {"evidence": "PC_FP32_ONNX_SOURCE_TIME_CONFIGURATION_SIMULATION_NOT_RK3566_INT8",
               "source_video": str(args.video.resolve()), "source_sha256": digest(args.video),
               "source_frames": total, "processed_frames": processed, "source_fps": fps,
               "duration_seconds": processed/fps, "output_size": list(out_size), "loops": 0,
               "main_model": str(args.main_model.resolve()), "main_sha256": digest(args.main_model),
               "specialist_model": str(args.specialist_model.resolve()), "specialist_sha256": digest(args.specialist_model),
               "main_input": [512, 288], "specialist_input": [512, 192], "main_confidence": .25, "specialist_confidence": .05,
               "main_nms_iou": .45, "specialist_nms_iou": .70,
               "main_target_hz": args.main_fps, "specialist_target_hz": args.specialist_fps,
               "main_inference_updates": len(main_ms), "specialist_inference_updates": len(spec_ms),
               "main_max_source_age_ms": main_max_age, "specialist_max_source_age_ms": spec_max_age,
               "specialist_roi": "adaptive_top60_or_focus_one_forward_per_refresh", "roi_updates": dict(roi_counts),
               "main_fresh_box_counts_not_unique_tracks": dict(class_counts), "signal_display_frame_counts": dict(color_counts),
               "pc_preprocess_inference_decode_ms": {"main": statistics(main_ms), "specialist": statistics(spec_ms)},
               "wall_seconds": time.perf_counter()-started,
               "limitations": ["FP32 exports do not reproduce RKNN INT8 quantization errors or board latency",
                               "Single front stream; no PCIe/FPGA/capture/display hardware simulation",
                               "Between updates boxes hold source coordinates; no tracking or ego motion",
                               "No lane model or synthetic lane/navigation/speed overlays",
                               "Signal colour/shape heuristics are experimental; ego signal association unverified",
                               "Unlabelled video; no detection accuracy metric claimed"]}
    (args.output_dir/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
