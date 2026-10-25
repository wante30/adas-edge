"""B-05O: simulate 10 Hz / 5 Hz video-time schedules on official videos.

This is a PC scheduling simulation, not RK3566 performance measurement.  At each
display frame only results from the latest past inference are shown.  No future
frame is used and no tracker interpolates detections.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_ai.traffic_light_tracking import classify_lamp_color  # noqa: E402
from tools.b05k_common import box_iou, decode_yolo11, prepare_input  # noqa: E402
from tools.b05n_tl_eval import decode_class, prepare_full  # noqa: E402

CLASS_BGR = {
    "person": (0, 200, 255), "rider": (255, 200, 0), "car": (0, 255, 0), "truck": (0, 128, 255),
    "bus": (255, 0, 255), "train": (128, 0, 255), "motor": (255, 128, 0), "bike": (0, 255, 255),
    "traffic light": (0, 0, 255), "traffic sign": (255, 255, 0),
}
TL_COLOR = {"RED": (40, 40, 235), "GREEN": (40, 210, 60), "YELLOW": (30, 220, 240), "UNKNOWN": (230, 230, 230)}
SCHEMES = ("bdd_10hz", "p2_10hz", "dual_10_5hz")
MAIN_HZ = 10.0
SPEC_HZ = 5.0
TITLES = {
    "bdd_10hz": "BDD-YOLOv8n 10Hz",
    "p2_10hz": "YOLOv8n-P2 10-class 10Hz",
    "dual_10_5hz": "BDD 10Hz + TL Specialist 5Hz (union)",
}


def bucket(t: float, hz: float) -> int:
    """Time bucket index for a video timestamp; a new bucket means a new inference."""
    return int(np.floor(t * hz + 1e-6))


def full_frame_detections(model: Any, frame: np.ndarray, *, conf: float, iou: float, skip_tl: bool, t: float, source: str) -> list[dict]:
    import torch
    tensor, transform = prepare_full(frame, 512, 288)
    with torch.inference_mode():
        raw = model(torch.from_numpy(tensor).cuda())
    output = []
    for class_id, name in model.names.items():
        if int(class_id) == 8 and skip_tl:
            continue
        for item in decode_class(raw, int(class_id), conf, iou):
            x1, y1, x2, y2 = item["bbox"]
            item["bbox"] = [x1 / transform["scale_x"], y1 / transform["scale_y"], x2 / transform["scale_x"], y2 / transform["scale_y"]]
            item["class_id"] = int(class_id)
            item["class_name"] = str(name)
            item["t_infer"] = t
            item["source"] = source
            if name == "traffic light":
                obs = classify_lamp_color(frame, item["bbox"])
                item["color"] = obs.color.value
                item["color_valid"] = bool(obs.valid)
            output.append(item)
    return output


def specialist_detections(model: Any, frame: np.ndarray, *, conf: float, iou: float, t: float, source: str) -> list[dict]:
    import torch
    tensor, transform = prepare_input(frame, input_width=512, input_height=192, roi_top=0.60, preprocess="resize")
    with torch.inference_mode():
        raw = model(torch.from_numpy(tensor).cuda())
    output = []
    for item in decode_yolo11(raw, conf=conf, iou=iou):
        item["bbox"] = transform.network_to_frame(item["bbox"])
        obs = classify_lamp_color(frame, item["bbox"])
        item.update({"class_id": 8, "class_name": "traffic light", "color": obs.color.value,
                     "color_valid": bool(obs.valid), "t_infer": t, "source": source})
        output.append(item)
    return output


def fuse_tl(bdd_dets: list[dict], spec_dets: list[dict], iou_thr: float = 0.5) -> list[dict]:
    """Union of BDD and specialist TL boxes with NMS, mirroring tools/b05c_diagnose.fuse."""
    candidates = [d for d in bdd_dets if d.get("class_name") == "traffic light"] + list(spec_dets)
    candidates.sort(key=lambda det: -det["confidence"])
    kept: list[dict] = []
    for det in candidates:
        if any(box_iou(det["bbox"], other["bbox"]) >= iou_thr for other in kept):
            continue
        kept.append(det)
    return kept


def draw(frame: np.ndarray, detections: list[dict], *, title: str, t: float, mode: str, stats: str) -> np.ndarray:
    canvas = frame.copy()
    for item in detections:
        if mode == "tl" and item.get("class_name") != "traffic light":
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in item["bbox"])
        age = max(0.0, (t - float(item.get("t_infer", t))) * 1000.0)
        if item.get("class_name") == "traffic light":
            color = TL_COLOR.get(item.get("color", "UNKNOWN"), TL_COLOR["UNKNOWN"])
            label = f"{item.get('color', 'UNKNOWN')} {item['confidence']:.2f}"
        else:
            color = CLASS_BGR.get(item["class_name"], (220, 220, 220))
            label = f"{item['class_name']} {item['confidence']:.2f}"
        if age > 0.5:
            label += f" held{age:.0f}ms"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2 if age < 0.5 else 1)
        cv2.putText(canvas, label, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1 if age >= 0.5 else 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 46), (18, 24, 28), -1)
    cv2.putText(canvas, f"{title}   {stats}", (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 248, 250), 2, cv2.LINE_AA)
    return canvas


def compose(panels: dict[str, np.ndarray], title: str, stats_text: str) -> np.ndarray:
    cell_w, cell_h = 960, 540
    canvas = np.full((1080, 1920, 3), 24, np.uint8)
    positions = {"bdd_10hz": (0, 0), "p2_10hz": (960, 0), "dual_10_5hz": (0, 540)}
    for label, panel in panels.items():
        if label not in positions:
            continue
        x, y = positions[label]
        canvas[y:y + cell_h, x:x + cell_w] = cv2.resize(panel, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    x, y = 960, 540
    cv2.rectangle(canvas, (x, y), (1919, 1079), (16, 20, 24), -1)
    cv2.putText(canvas, "SCHEDULE SIMULATION", (x + 22, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 245, 248), 2, cv2.LINE_AA)
    for i, line in enumerate(stats_text.split("\n")):
        cv2.putText(canvas, line, (x + 22, y + 95 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (190, 220, 230), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (1919, 34), (12, 16, 18), -1)
    cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 248, 250), 1, cv2.LINE_AA)
    return canvas


def track_metrics(events: list[dict], source: str) -> dict[str, Any]:
    """Greedy IoU association on fresh (not held) detections only."""
    finished: list[dict] = []
    active: list[dict] = []
    for event in events:
        dets = [d for d in event["detections"] if d.get("source") == source]
        used: set[int] = set()
        for det in sorted(dets, key=lambda x: -x["confidence"]):
            best, best_iou = None, 0.0
            for index, track in enumerate(active):
                if index in used:
                    continue
                overlap = box_iou(det["bbox"], track["last_bbox"])
                if overlap > best_iou:
                    best_iou, best = overlap, index
            if best is not None and best_iou >= 0.5:
                used.add(best)
                track = active[best]
                track["last_bbox"] = det["bbox"]
                track["end"] = event["t"]
                track["updates"] += 1
                track["ious"].append(best_iou)
            else:
                used.add(len(active))
                active.append({"start": event["t"], "end": event["t"], "last_bbox": det["bbox"],
                               "updates": 1, "ious": []})
        still = []
        for track in active:
            (still if (event["t"] - track["end"]) <= 1.0 else finished).append(track)
        active = still
    tracks = finished + active
    durations = [t["end"] - t["start"] for t in tracks]
    ious = [v for t in tracks for v in t["ious"]]
    return {
        "source": source,
        "tracks": len(tracks),
        "mean_duration_s": float(np.mean(durations)) if durations else None,
        "p50_duration_s": float(np.percentile(durations, 50)) if durations else None,
        "short_tracks_1_update": sum(1 for t in tracks if t["updates"] <= 1),
        "short_track_fraction": sum(1 for t in tracks if t["updates"] <= 1) / len(tracks) if tracks else 0.0,
        "mean_updates_per_track": float(np.mean([t["updates"] for t in tracks])) if tracks else None,
        "mean_iou_between_updates": float(np.mean(ious)) if ious else None,
        "p05_iou_between_updates": float(np.percentile(ious, 5)) if ious else None,
    }


def process(video: Path, bdd_path: Path, spec_path: Path, p2_path: Path, out: Path, *, conf: float, iou: float, key_times: list[float], upper: bool, fusion: str = "union") -> dict[str, Any]:
    from ultralytics import YOLO
    import torch

    mode_name = "upper_bound" if upper else "scheduled"
    out_dir = out / video.stem / mode_name
    out_dir.mkdir(parents=True, exist_ok=True)
    bdd = YOLO(str(bdd_path)).model.cuda().eval()
    spec = YOLO(str(spec_path)).model.cuda().eval()
    p2 = YOLO(str(p2_path)).model.cuda().eval()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    effective_schemes = SCHEMES if not upper else ("bdd_10hz", "p2_10hz", "dual_10_5hz")
    writers = {}
    comp_writers = {}
    for scheme in effective_schemes:
        for mode in ("tl", "all"):
            writers[(scheme, mode)] = cv2.VideoWriter(str(out_dir / f"{scheme}_{mode}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for mode in ("tl", "all"):
        comp_writers[mode] = cv2.VideoWriter(str(out_dir / f"compare_{mode}_3schemes.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))

    latest = {"bdd": [], "p2": [], "spec": []}
    infer_times = {"bdd": [], "p2": [], "spec": []}
    events = []
    count = {"bdd": 0, "p2": 0, "spec": 0}
    last_main_bucket = -1
    last_spec_bucket = -1
    frame_count = 0
    display_stats = {scheme: {"tl_frames": 0, "any_frames": 0, "first_tl_s": None} for scheme in SCHEMES}
    key_events = []
    started = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        t = frame_count / fps
        if upper:
            main_due = spec_due = True
        else:
            main_bucket = bucket(t, MAIN_HZ)
            spec_bucket = bucket(t, SPEC_HZ)
            main_due = main_bucket != last_main_bucket
            spec_due = spec_bucket != last_spec_bucket
            if main_due:
                last_main_bucket = main_bucket
            if spec_due:
                last_spec_bucket = spec_bucket
        if main_due:
            t0 = time.perf_counter()
            latest["bdd"] = full_frame_detections(bdd, frame, conf=conf, iou=iou, skip_tl=False, t=t, source="bdd")
            infer_times["bdd"].append((time.perf_counter() - t0) * 1000.0)
            count["bdd"] += 1
            t0 = time.perf_counter()
            latest["p2"] = full_frame_detections(p2, frame, conf=conf, iou=iou, skip_tl=False, t=t, source="p2")
            infer_times["p2"].append((time.perf_counter() - t0) * 1000.0)
            count["p2"] += 1
        if spec_due:
            t0 = time.perf_counter()
            latest["spec"] = specialist_detections(spec, frame, conf=conf, iou=iou, t=t, source="spec")
            infer_times["spec"].append((time.perf_counter() - t0) * 1000.0)
            count["spec"] += 1
        tl_events = [d for d in (latest["bdd"] + latest["p2"] + latest["spec"])
                     if d.get("class_name") == "traffic light" and abs(float(d.get("t_infer", -1.0)) - t) < 1e-6]
        if main_due or spec_due:
            events.append({"t": round(t, 4), "frame": frame_count, "main": main_due, "spec": spec_due,
                           "detections": tl_events})
        bdd_tl = [d for d in latest["bdd"] if d.get("class_name") == "traffic light"]
        dual_tl = fuse_tl(bdd_tl, latest["spec"], iou) if fusion == "union" else list(latest["spec"])
        schemes = {
            "bdd_10hz": latest["bdd"],
            "p2_10hz": latest["p2"],
            "dual_10_5hz": [d for d in latest["bdd"] if d.get("class_name") != "traffic light"] + dual_tl,
        }
        prefix = "UPPER 30Hz " if upper else ""
        elapsed = t if t > 1e-6 else 0.0
        rate = {key: (count[key] - 1) / elapsed if elapsed > 0 else 0.0 for key in ("bdd", "p2", "spec")}
        dual_total = rate["bdd"] + rate["spec"]
        scheme_stats = {
            "bdd_10hz": f"fwd {count['bdd']}  {rate['bdd']:.2f} Hz",
            "p2_10hz": f"fwd {count['p2']}  {rate['p2']:.2f} Hz",
            "dual_10_5hz": f"BDD {rate['bdd']:.2f}Hz + SPEC {rate['spec']:.2f}Hz = {dual_total:.2f} fwd/s",
        }
        stats_lines = [f"{video.stem}", f"t={t:6.2f}s frame={frame_count:06d}",
                       "target: main 10Hz, spec 5Hz" if not upper else "offline upper bound 30Hz (each model)",
                       f"BDD {rate['bdd']:.2f} Hz  fwd {count['bdd']}",
                       f"P2  {rate['p2']:.2f} Hz  fwd {count['p2']}",
                       f"SPEC {rate['spec']:.2f} Hz  fwd {count['spec']}",
                       f"DUAL total {dual_total:.2f} fwd/s"]
        tl_panels, all_panels = {}, {}
        is_key = any(abs(t - target) <= 0.5 / fps for target in key_times)
        for scheme, dets in schemes.items():
            tl_dets = [d for d in dets if d.get("class_name") == "traffic light"]
            stats = scheme_stats[scheme]
            tl_panel = draw(frame, dets, title=f"{prefix}{TITLES[scheme]} TL-only", t=t, mode="tl", stats=stats)
            all_panel = draw(frame, dets, title=f"{prefix}{TITLES[scheme]} all classes", t=t, mode="all", stats=stats)
            writers[(scheme, "tl")].write(tl_panel)
            writers[(scheme, "all")].write(all_panel)
            tl_panels[scheme] = tl_panel
            all_panels[scheme] = all_panel
            if tl_dets:
                display_stats[scheme]["tl_frames"] += 1
                if display_stats[scheme]["first_tl_s"] is None:
                    display_stats[scheme]["first_tl_s"] = round(t, 3)
            if dets:
                display_stats[scheme]["any_frames"] += 1
            if is_key:
                key_events.append({"t": round(t, 3), "scheme": scheme, "tl_count": len(tl_dets),
                                   "detections": tl_dets})
                cv2.imwrite(str(out_dir / f"{scheme}_t{t:07.3f}.jpg"), all_panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        comp_writers["tl"].write(compose(tl_panels, f"{video.stem} traffic-light only | 10/5Hz schedule" if not upper else f"{video.stem} TL | 30Hz upper bound", "\n".join(stats_lines)))
        comp_writers["all"].write(compose(all_panels, f"{video.stem} all classes | 10/5Hz schedule" if not upper else f"{video.stem} all classes | 30Hz upper bound", "\n".join(stats_lines)))
        frame_count += 1
        if frame_count % 300 == 0:
            print(f"{video.stem} {mode_name} frame {frame_count} {t:.1f}s", flush=True)
    capture.release()
    for writer in writers.values():
        writer.release()
    for writer in comp_writers.values():
        writer.release()
    duration = frame_count / fps
    rates = {key: {"count": count[key], "hz": count[key] / duration if duration else 0.0,
                   "mean_ms": statistics.fmean(infer_times[key]) if infer_times[key] else None,
                   "p95_ms": float(np.percentile(infer_times[key], 95)) if infer_times[key] else None}
             for key in ("bdd", "p2", "spec")}
    scheme_rates = {
        "bdd_10hz": rates["bdd"]["count"] / duration,
        "p2_10hz": rates["p2"]["count"] / duration,
        "dual_10_5hz": (rates["bdd"]["count"] + rates["spec"]["count"]) / duration,
    }
    metrics = {
        "video": str(video.resolve()), "mode": mode_name, "fusion": fusion,
        "frames": frame_count, "fps": fps, "duration_s": duration,
        "target": {"bdd_hz": MAIN_HZ, "p2_hz": MAIN_HZ, "specialist_hz": SPEC_HZ} if not upper else {"all_hz": fps},
        "rates": rates, "scheme_total_forward_per_second": scheme_rates,
        "tl_tracks": {key: track_metrics(events, key) for key in ("bdd", "p2", "spec")},
        "display_stats": display_stats,
        "key_events": key_events,
        "events": events,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--bdd", type=Path, required=True)
    ap.add_argument("--specialist", type=Path, required=True)
    ap.add_argument("--p2", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--key-times", type=float, nargs="*", default=[])
    ap.add_argument("--upper", action="store_true")
    ap.add_argument("--fusion", choices=("union", "replace"), default="union")
    args = ap.parse_args()
    print(json.dumps(process(args.video, args.bdd, args.specialist, args.p2, args.output,
                             conf=args.conf, iou=args.iou, key_times=args.key_times, upper=args.upper,
                             fusion=args.fusion), indent=2))


if __name__ == "__main__":
    main()
