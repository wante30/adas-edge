"""Explain sparse real-video traffic-light misses by pipeline stage.

The script keeps a low-score candidate trace, so a missed annotation can be
classified as model/no-candidate, threshold, NMS, coordinate mapping, or
tracker/TTL loss.  It never changes the default online threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from competition_ai.multi_signal import Candidate, MultiSignalTracker, iou
from tools.offline_signal_models import OfflineDetector, sha256


SPECS = {
    "bdd": ("rknn_b05o/bdd_yolov8n_512x288_raw_heads.onnx", False, 8),
    "p2": ("rknn_b05o/b05n_p2_multiclass_512x288_raw_heads_20261026.onnx", False, 8),
    "specialist": ("rknn_b05o/b05l_v8n_p2_512x192_raw_heads_20261026.onnx", False, 0),
}


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def intersection_area(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))


def coverage(box, target):
    area = max(0., target[2]-target[0]) * max(0., target[3]-target[1])
    return intersection_area(box, target) / max(area, 1e-6)


def best(rows, target):
    if not rows:
        return {"iou": 0., "coverage": 0., "confidence": 0., "bbox": None}
    row = max(rows, key=lambda item: (iou(item["bbox"], target), coverage(item["bbox"], target)))
    return {"iou": iou(row["bbox"], target), "coverage": coverage(row["bbox"], target),
            "confidence": row["confidence"], "bbox": row["bbox"],
            "source_index": row.get("source_index")}


def stage_for(trace, target, lamp_target, threshold):
    pre = trace.get("pre_nms", [])
    post = trace.get("post_nms", [])
    filtered = [row for row in post if row["confidence"] >= threshold]
    pre_best = best(pre, target)
    post_best = best(post, target)
    model_candidates = [row for row in pre if iou(row["bbox"], target) >= .5 or coverage(row["bbox"], lamp_target) >= .5]
    post_candidates = [row for row in post if iou(row["bbox"], target) >= .5 or coverage(row["bbox"], lamp_target) >= .5]
    accepted_candidates = [row for row in filtered if iou(row["bbox"], target) >= .5 or coverage(row["bbox"], lamp_target) >= .5]
    accepted_best = best(accepted_candidates, target)
    mapping_model_match = False
    mapping_details = None
    params = trace.get("letterbox", {})
    scale = float(params.get("scale", 1.0)); px = float(params.get("pad_x", 0)); py = float(params.get("pad_y", 0))
    if scale > 0:
        model_target = [target[0]*scale+px, target[1]*scale+py, target[2]*scale+px, target[3]*scale+py]
        model_rows = [{**row, "bbox": row.get("bbox_model", row["bbox"])} for row in pre]
        model_best = best(model_rows, model_target)
        mapping_model_match = model_best["iou"] >= .5 or model_best["coverage"] >= .5
        mapping_details = {"target_model": model_target, "best_model": model_best}
    if model_candidates and not post_candidates:
        stage = "NMS_SUPPRESSED"
    elif post_candidates and not accepted_candidates:
        stage = "CONF_THRESHOLD"
    elif mapping_model_match and not model_candidates:
        stage = "COORDINATE_MAPPING"
    elif not model_candidates:
        small = max(lamp_target[2]-lamp_target[0], lamp_target[3]-lamp_target[1]) <= 10
        stage = "SMALL_TARGET_OR_MODEL" if small else "MODEL_NO_CANDIDATE"
    else:
        stage = "ACCEPTED_OR_PARTIAL_BOX"
    return {"stage": stage, "threshold": threshold, "pre_nms": pre_best, "post_nms": post_best,
            "accepted": accepted_best, "pre_match": bool(model_candidates),
            "post_match": bool(post_candidates), "accepted_match": bool(accepted_candidates),
            "mapping": mapping_details, "nms_suppressed_count": len(trace.get("nms_suppressed", []))}


def mapped_candidates(trace, threshold):
    return [row for row in trace.get("post_nms", []) if row["confidence"] >= threshold]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT/"docs/multi_signal/offline_eval_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips", nargs="*", default=[])
    parser.add_argument("--threshold", type=float, default=.25)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    screenshots = args.output/"screenshots"; screenshots.mkdir(exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    model_root = Path(manifest["model_root"])
    detectors = {name: OfflineDetector(model_root/spec[0], logits=spec[1], traffic_light_id=spec[2])
                 for name, spec in SPECS.items()}
    dump(args.output/"models.json", {name: detector.evidence for name, detector in detectors.items()})
    case_rows, stage_rows = [], []
    overall = Counter()
    for clip in manifest["videos"]:
        if args.clips and clip["id"] not in args.clips:
            continue
        cap = cv2.VideoCapture(clip["path"])
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {clip['path']}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.)
        labels = {int(row["frame_id"]): row for row in clip.get("labels", [])}
        trackers = {name: MultiSignalTracker(confirm_updates=2) for name in ("bdd", "p2", "specialist", "p2_specialist")}
        fid = 0
        while True:
            ok, original = cap.read()
            if not ok:
                break
            q4 = cv2.resize(original, (640, 360), interpolation=cv2.INTER_AREA)
            sx, sy = original.shape[1]/640., original.shape[0]/360.
            ns = 1_000_000_000 + round(fid/fps*1e9)
            if fid % max(1, args.stride) == 0 or fid in labels:
                traces = {}
                for name, detector in detectors.items():
                    inp = q4[:216] if name == "specialist" else q4
                    _, _, trace = detector.detect(inp, return_trace=True)
                    if name == "specialist":
                        # Crop origin is (0,0), so specialist coordinates already
                        # live in Q4 native space; only clip to the full height.
                        for group in ("pre_nms", "post_nms", "nms_suppressed"):
                            for row in trace[group]:
                                row["bbox"][1] = max(0., min(360., row["bbox"][1]))
                                row["bbox"][3] = max(0., min(360., row["bbox"][3]))
                    traces[name] = trace
                current_frame = fid
            else:
                current_frame = None
            if current_frame == fid:
                for name in ("bdd", "p2", "specialist"):
                    rows = mapped_candidates(traces[name], args.threshold)
                    candidates = [Candidate(tuple(row["bbox"]), row["confidence"], fid, ns, name) for row in rows]
                    trackers[name].update(q4, candidates, frame_id=fid, timestamp_ns=ns,
                                          context=("STAGE3", clip["id"], name))
                union_rows = mapped_candidates(traces["p2"], args.threshold) + mapped_candidates(traces["specialist"], args.threshold)
                trackers["p2_specialist"].update(
                    q4, [Candidate(tuple(row["bbox"]), row["confidence"], fid, ns, detector)
                         for detector, rows in (("p2", mapped_candidates(traces["p2"], args.threshold)),
                                                ("specialist", mapped_candidates(traces["specialist"], args.threshold)))
                         for row in rows],
                    frame_id=fid, timestamp_ns=ns, context=("STAGE3", clip["id"], "p2_specialist"))
            if fid in labels:
                truth_rows = []
                for object_id, truth in enumerate(labels[fid]["objects"]):
                    gt = [value/(sx if index % 2 == 0 else sy) for index, value in enumerate(truth["bbox_native"])]
                    lamp = [value/(sx if index % 2 == 0 else sy) for index, value in enumerate(truth.get("lamp_bbox_native", truth["bbox_native"]))]
                    per_model = {}
                    for name in ("bdd", "p2", "specialist"):
                        detail = stage_for(traces[name], gt, lamp, args.threshold)
                        per_model[name] = detail
                        stage_rows.append({"clip": clip["id"], "frame_id": fid, "object_id": object_id,
                                           "model": name, "stage": detail["stage"],
                                           "best_pre_iou": detail["pre_nms"]["iou"],
                                           "best_post_iou": detail["post_nms"]["iou"],
                                           "best_accepted_iou": detail["accepted"]["iou"],
                                           "best_pre_coverage": detail["pre_nms"]["coverage"],
                                           "best_accepted_confidence": detail["accepted"]["confidence"]})
                        overall[f"{name}:{detail['stage']}"] += 1
                    union = []
                    for name in ("p2", "specialist"):
                        union.extend(mapped_candidates(traces[name], args.threshold))
                    union_best = best(union, gt)
                    snapshot = trackers["p2_specialist"].snapshot(ns)
                    matched_tracks = [signal for signal in snapshot["signals"] if iou(signal["bbox"], gt) >= .5 or coverage(signal["bbox"], lamp) >= .5]
                    union_detail = {"candidate": union_best, "track_matches": matched_tracks,
                                    "track_valid": any(signal.get("valid") for signal in matched_tracks),
                                    "rejections": snapshot.get("rejections", []),
                                    "stage": "TRACK_VALID" if any(signal.get("valid") for signal in matched_tracks)
                                    else "TRACK_LOST_OR_UNCONFIRMED" if union else "NO_FUSED_CANDIDATE"}
                    overall[f"p2_specialist:{union_detail['stage']}"] += 1
                    entry = {"clip": clip["id"], "frame_id": fid, "object_id": object_id,
                             "truth": truth, "bbox_q4": gt, "lamp_bbox_q4": lamp,
                             "models": per_model, "p2_specialist": union_detail,
                             "preprocess_contract": {name: traces[name]["letterbox"] for name in traces}}
                    case_rows.append(entry)
                    tile = original.copy()
                    x1, y1, x2, y2 = map(round, [gt[0]*sx, gt[1]*sy, gt[2]*sx, gt[3]*sy])
                    cv2.rectangle(tile, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(tile, f"GT {clip['id']} f{fid} obj{object_id}", (x1, max(20, y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
                    colors = {"bdd": (255, 80, 80), "p2": (80, 220, 80), "specialist": (80, 160, 255)}
                    for name, detail in per_model.items():
                        row = detail["accepted"].get("bbox")
                        if row:
                            bx = [round(row[0]*sx), round(row[1]*sy), round(row[2]*sx), round(row[3]*sy)]
                            cv2.rectangle(tile, tuple(bx[:2]), tuple(bx[2:]), colors[name], 2)
                            cv2.putText(tile, f"{name} {detail['stage']} {detail['accepted']['confidence']:.2f}",
                                        (bx[0], min(original.shape[0]-10, bx[3]+18)), cv2.FONT_HERSHEY_SIMPLEX, .5, colors[name], 1)
                    cv2.imwrite(str(screenshots/f"{clip['id']}_{fid:06d}_{object_id}.png"), tile)
            fid += 1
        cap.release()
    with (args.output/"per_case.jsonl").open("w", encoding="utf-8") as stream:
        for row in case_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    fields = list(stage_rows[0]) if stage_rows else []
    if fields:
        with (args.output/"stage_cases.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(stage_rows)
    dump(args.output/"stage_summary.json", {"threshold": args.threshold, "cases": len(case_rows),
                                             "counts": dict(overall), "video_sha256": {
                                                 clip["id"]: sha256(clip["path"]) for clip in manifest["videos"]
                                                 if not args.clips or clip["id"] in args.clips}})
    print(json.dumps({"cases": len(case_rows), "counts": dict(overall)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
