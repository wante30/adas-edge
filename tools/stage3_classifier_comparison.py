"""Compare traditional, Autoware CNN, and conservative fusion on the same boxes."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from collections import Counter
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from competition_ai.autoware_classifier import AutowareMobileNetV2Backend, ConservativeFusionBackend
from competition_ai.multi_signal import Candidate, MultiSignalTracker, iou
from competition_ai.signal_shape import ContourTemplateBackend
from tools.offline_signal_models import sha256


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def crop(image, box):
    x1, y1 = np.floor(box[:2]).astype(int); x2, y2 = np.ceil(box[2:]).astype(int)
    return image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]


def score(counts, prefix, truth, attr):
    for key in ("color", "shape"):
        wanted = truth.get(key, "UNKNOWN")
        if wanted == "UNKNOWN":
            continue
        got = attr.get(key, "UNKNOWN")
        counts[f"{prefix}{key}_total"] += 1
        counts[f"{prefix}{key}_correct"] += got == wanted
        counts[f"{prefix}{key}_unknown"] += got == "UNKNOWN"
        counts[f"{prefix}{key}_wrong_known"] += got not in (wanted, "UNKNOWN")
        if key == "shape" and wanted.endswith("_ARROW"):
            counts[f"{prefix}arrow_total"] += 1
            counts[f"{prefix}arrow_correct"] += got == wanted
            counts[f"{prefix}arrow_unknown"] += got == "UNKNOWN"


def attr_row(backend, roi):
    started = time.perf_counter()
    value = asdict(backend.classify(roi))
    value["wall_ms"] = (time.perf_counter() - started) * 1000.
    return value


def draw_signal(canvas, signal, *, sx=1., sy=1., offset=(0, 0)):
    x1, y1, x2, y2 = [round(signal["bbox"][index] * (sx if index % 2 == 0 else sy)) for index in range(4)]
    x1 += offset[0]; x2 += offset[0]; y1 += offset[1]; y2 += offset[1]
    color = {"RED": (40, 40, 255), "GREEN": (40, 230, 40), "YELLOW": (0, 230, 230)}.get(signal.get("color"), (170, 170, 170))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    cv2.putText(canvas, f"#{signal['track_id']} {signal.get('color')} {signal.get('shape')}",
                (max(0, x1), max(16, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1, cv2.LINE_AA)


def panel(image, result, title):
    canvas = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA).copy()
    for signal in result.get("signals", []):
        if signal.get("valid"):
            draw_signal(canvas, signal, sx=1., sy=1.)
    cv2.putText(canvas, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, "observed_directions: visual only / ego directions: UNKNOWN", (6, 350),
                cv2.FONT_HERSHEY_SIMPLEX, .33, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT/"docs/multi_signal/offline_eval_manifest.json")
    parser.add_argument("--cached-root", type=Path, default=ROOT/"outputs/offline_raw_export_final")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cnn-model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--clips", nargs="*", default=[])
    parser.add_argument("--threshold", type=float, default=.25)
    parser.add_argument("--attribute-hz", type=float, default=2.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    failures = args.output/"failures"; failures.mkdir()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    traditional = ContourTemplateBackend()
    cnn = AutowareMobileNetV2Backend(args.cnn_model, args.labels)
    fused = ConservativeFusionBackend(traditional, cnn)
    backends = {"traditional": traditional, "cnn": cnn, "hybrid": fused}
    reports = {}; all_counts = Counter(); timing = Counter(); timing_count = Counter()
    all_manual_rows = []
    csv_rows = []
    for clip in manifest["videos"]:
        if args.clips and clip["id"] not in args.clips:
            continue
        cached_path = args.cached_root/clip["id"]/"frames.jsonl"
        if not cached_path.is_file():
            raise FileNotFoundError(f"missing cached detections: {cached_path}")
        cached = {int(row["frame_id"]): row for row in map(json.loads, cached_path.read_text(encoding="utf-8").splitlines())}
        cap = cv2.VideoCapture(clip["path"])
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {clip['path']}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.)
        writers = {name: cv2.VideoWriter(str(args.output/f"{clip['id']}_{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 400))
                   for name in backends}
        comparison = cv2.VideoWriter(str(args.output/f"{clip['id']}_comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 400))
        trackers = {(name, resolution): MultiSignalTracker(backend=backend, confirm_updates=2)
                    for name, backend in backends.items() for resolution in ("q4", "original")}
        labels = {int(row["frame_id"]): row for row in clip.get("labels", [])}
        count = Counter(); rows_out = []
        fid = 0; next_update = 0
        while True:
            ok, original = cap.read()
            if not ok:
                break
            q4 = cv2.resize(original, (640, 360), interpolation=cv2.INTER_AREA)
            sx, sy = original.shape[1]/640., original.shape[0]/360.
            ns = 1_000_000_000 + round(fid/fps*1e9)
            cached_frame = cached.get(fid, {})
            raw_rows = cached_frame.get("raw_detections", {})
            union = []
            for detector in ("p2", "specialist"):
                for row in raw_rows.get(detector, []):
                    if row.get("confidence", 0.) >= args.threshold:
                        union.append((detector, row))
            due = fid >= next_update or fid in labels
            if due:
                for name, backend in backends.items():
                    candidates_q4 = [Candidate(tuple(row["bbox"]), float(row["confidence"]), fid, ns, detector)
                                     for detector, row in union]
                    candidates_native = [Candidate(tuple(value*(sx if index % 2 == 0 else sy) for index, value in enumerate(row["bbox"])),
                                                   float(row["confidence"]), fid, ns, detector)
                                         for detector, row in union]
                    trackers[name, "q4"].update(q4, candidates_q4, frame_id=fid, timestamp_ns=ns,
                                                 context=("STAGE3", clip["id"], "q4"))
                    trackers[name, "original"].update(original, candidates_native, frame_id=fid, timestamp_ns=ns,
                                                       context=("STAGE3", clip["id"], "original"))
                    next_update = fid + max(1, round(fps/max(args.attribute_hz, 1e-6)))
            results = {}
            for name in backends:
                results[name] = {}
                for resolution, now in (("q4", q4), ("original", original)):
                    result = trackers[name, resolution].snapshot(ns)
                    result["input_resolution"] = [int(now.shape[1]), int(now.shape[0])]
                    result["offline_coordinate_space"] = "q4_native" if resolution == "q4" else "source_video_native_reference"
                    if resolution == "original":
                        for signal in result["signals"]:
                            signal["bbox_space"] = "source_video_native_reference"
                    results[name][resolution] = result
            panels = {name: panel(q4, results[name]["q4"], f"{name} / P2+Specialist / q4") for name in backends}
            for name, writer in writers.items():
                tile = np.zeros((400, 640, 3), np.uint8); tile[:360] = panels[name]
                cv2.putText(tile, "native ROI is separately logged in JSONL", (6, 382), cv2.FONT_HERSHEY_SIMPLEX, .42, (210, 220, 230), 1)
                writer.write(tile)
            joined = np.zeros((400, 1920, 3), np.uint8)
            joined[:360, :640] = panels["traditional"]
            joined[:360, 640:1280] = panels["cnn"]
            joined[:360, 1280:] = panels["hybrid"]
            comparison.write(joined)
            reviewed = []
            if fid in labels:
                for object_id, truth in enumerate(labels[fid]["objects"]):
                    native_box = truth.get("attribute_roi_native", truth["bbox_native"])
                    q4_box = [value/(sx if index % 2 == 0 else sy) for index, value in enumerate(native_box)]
                    entry = {"frame_id": fid, "object_id": object_id, "truth": truth, "manual_roi": {}, "detected": {}}
                    for resolution, image, box in (("q4", q4, q4_box), ("original", original, native_box)):
                        entry["manual_roi"][resolution] = {}
                        for name, backend in backends.items():
                            attr = attr_row(backend, crop(image, box))
                            entry["manual_roi"][resolution][name] = attr
                            score(count, f"manual_{resolution}_{name}_", truth, attr)
                            timing[name] += attr.get("wall_ms", 0.); timing_count[name] += 1
                            if attr.get("color") != truth.get("color") or attr.get("shape") != truth.get("shape"):
                                stem = f"{clip['id']}_{fid:06d}_{object_id}_{resolution}_{name}"
                                cv2.imwrite(str(failures/(stem+".png")), crop(image, box))
                                dump(failures/(stem+".json"), {"truth": truth, "attributes": attr, "resolution": resolution})
                    for name in backends:
                        signal = max(results[name]["original"]["signals"], key=lambda row: iou(row["bbox"], native_box), default=None)
                        entry["detected"][name] = {"matched": bool(signal and iou(signal["bbox"], native_box) >= .5),
                                                    "signal": signal}
                    reviewed.append(entry)
                rows_out.extend(reviewed)
                count["reviewed_frames"] += 1; count["gt_objects"] += len(labels[fid]["objects"])
            for name in backends:
                for resolution in ("q4", "original"):
                    result = results[name][resolution]
                    for signal in result["signals"]:
                        if signal.get("valid"):
                            count[f"tracks_{name}_{resolution}"] += 1
            with (args.output/f"{clip['id']}_frames.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"frame_id": fid, "timestamp_ms": fid/fps*1000.,
                                         "raw_detections": raw_rows, "pipelines": results,
                                         "reviewed": reviewed}, ensure_ascii=False, allow_nan=False)+"\n")
            fid += 1
        cap.release()
        for writer in writers.values(): writer.release()
        comparison.release()
        report = {"clip": clip["id"], "video": clip["path"], "source_sha256": sha256(clip["path"]),
                  "frames": fid, "counts": dict(count), "annotation_scope": "SPARSE_MANUAL_ROIS",
                  "attribute_hz": args.attribute_hz,
                  "detector_source": str(cached_path),
                  "backends": list(backends)}
        dump(args.output/f"{clip['id']}_report.json", report)
        reports[clip["id"]] = report; all_counts.update(count)
        all_manual_rows.extend(rows_out)
        csv_rows.extend({"clip": clip["id"], "metric": key, "value": value} for key, value in count.items())
    dump(args.output/"manual_roi_results.json", all_manual_rows)
    dump(args.output/"summary.json", {"reports": reports, "counts": dict(all_counts),
                                       "latency_ms_mean": {name: timing[name]/max(1, timing_count[name]) for name in backends},
                                       "latency_samples": dict(timing_count),
                                       "cnn_model": str(args.cnn_model), "cnn_labels": str(args.labels)})
    if csv_rows:
        with (args.output/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["clip", "metric", "value"]); writer.writeheader(); writer.writerows(csv_rows)
    print(json.dumps({"clips": list(reports), "counts": dict(all_counts),
                      "latency_ms_mean": {name: timing[name]/max(1, timing_count[name]) for name in backends}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
