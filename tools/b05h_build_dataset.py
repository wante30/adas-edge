"""B-05H build an auto-labelled traffic-light dataset from teacher predictions.

A: Specialist>=0.30 and BDD consensus (or Specialist>=0.50)
B: candidate but insufficient consensus/quality/colour
C: blue-sign-like false candidate with no trusted light

Human BDD GT is indexed separately and is never mixed into pseudo labels.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from competition_ai.traffic_light_tracking import classify_lamp_color


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0., min(ax2, bx2) - max(ax1, bx1)) * max(0., min(ay2, by2) - max(ay1, by1))
    aa = max(0., ax2 - ax1) * max(0., ay2 - ay1)
    bb = max(0., bx2 - bx1) * max(0., by2 - by1)
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.


def blue_ratio(frame, box):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    roi = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if roi.size == 0:
        return 0.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return float(((h >= 90) & (h <= 135) & (s >= 80) & (v >= 40)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    index = {Path(row["image"]).name: row for row in json.loads(Path(args.index).read_text(encoding="utf-8"))}
    out = Path(args.out)
    for sub in ("images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"):
        (out / "pseudo" / sub).mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "real_gt").mkdir(parents=True, exist_ok=True)

    rows = []
    counts = Counter()
    color_counts = Counter()
    size_counts = Counter()
    for image_path in sorted(Path(args.frames).rglob("*.jpg")):
        name = image_path.name
        if name not in index or name not in data["bdd"]:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        spec = data["specialist"].get(name, [])
        bdd = data["bdd"].get(name, [])
        used_spec = set()
        selected = []
        # High-confidence specialist boxes, with BDD consensus or very strong
        # specialist confidence, are trusted pseudo boxes.
        for s in sorted(spec, key=lambda x: x["confidence"], reverse=True):
            if s["confidence"] < 0.30:
                continue
            best_i, best_iou = None, 0.
            for i, b in enumerate(bdd):
                if i in used_spec:
                    continue
                overlap = iou(s["bbox"], b["bbox"])
                if overlap > best_iou:
                    best_i, best_iou = i, overlap
            consensus = best_i is not None and best_iou >= 0.30 and bdd[best_i]["confidence"] >= 0.10
            if not consensus and s["confidence"] < 0.50:
                continue
            if best_i is not None and best_iou >= 0.30:
                used_spec.add(best_i)
            box = [max(0., min(float(w), s["bbox"][0])), max(0., min(float(h), s["bbox"][1])),
                   max(0., min(float(w), s["bbox"][2])), max(0., min(float(h), s["bbox"][3]))]
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            selected.append({"bbox": box, "conf": s["confidence"], "class": "A",
                             "teacher": "specialist+bdd_consensus" if consensus else "specialist_strong"})
        # Pending candidates: any mid-confidence specialist or BDD-only boxes.
        for s in spec:
            if s["confidence"] < 0.15:
                continue
            if any(iou(s["bbox"], x["bbox"]) >= 0.5 for x in selected):
                continue
            box = [float(v) for v in s["bbox"]]
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            selected.append({"bbox": box, "conf": s["confidence"], "class": "B",
                             "teacher": "specialist_pending"})
        for b in bdd:
            if b["confidence"] < 0.25:
                continue
            if any(iou(b["bbox"], x["bbox"]) >= 0.5 for x in selected):
                continue
            selected.append({"bbox": [float(v) for v in b["bbox"]], "conf": b["confidence"],
                             "class": "B", "teacher": "bdd_pending"})
        # A blue-sign-like candidate with no trusted box is C.
        for s in spec:
            if s["confidence"] < 0.15 or any(iou(s["bbox"], x["bbox"]) >= 0.5 for x in selected):
                continue
            if blue_ratio(frame, s["bbox"]) >= 0.35:
                selected.append({"bbox": [float(v) for v in s["bbox"]], "conf": s["confidence"],
                                 "class": "C", "teacher": "blue_sign_reject"})
        # Deduplicate selected boxes, preferring A.
        dedup = []
        for item in sorted(selected, key=lambda x: (x["class"] != "A", -x["conf"])):
            if any(iou(item["bbox"], old["bbox"]) >= 0.6 for old in dedup):
                continue
            dedup.append(item)
        train_rows = []
        for item in dedup:
            x1, y1, x2, y2 = item["bbox"]
            if x2 <= x1 or y2 <= y1:
                continue
            color = classify_lamp_color(frame, (x1, y1, x2, y2))
            color_name = color.color.value if color.valid else "UNKNOWN"
            width = x2 - x1
            size = "<8" if width < 8 else "8-16" if width < 16 else "16-32" if width < 32 else ">32"
            counts[item["class"]] += 1
            color_counts[color_name] += 1
            size_counts[size] += 1
            train_rows.append({
                "image": name, "video": index[name]["video"],
                "timestamp_s": index[name]["timestamp_s"], "frame_id": index[name]["source_frame_id"],
                "bbox_xyxy": [round(v, 2) for v in item["bbox"]],
                "class_name": "traffic light", "class_id": 0,
                "color": color_name, "color_confidence": color.confidence,
                "teacher": item["teacher"], "confidence": round(item["conf"], 4),
                "review": item["class"], "source": "competition_video_pseudo",
            })
        # Split by source video, never by adjacent frame.
        video = index[name]["video"]
        split = "test" if video.startswith("video_20260929_113421") else (
            "val" if video.startswith("video_20260929_112739") else "train")
        label_path = out / "pseudo" / "labels" / split / f"{name.rsplit('.', 1)[0]}.txt"
        image_out = out / "pseudo" / "images" / split / name
        shutil.copy2(image_path, image_out)
        if train_rows:
            with label_path.open("w", encoding="utf-8") as fh:
                for item in train_rows:
                    if item["review"] == "C":
                        continue
                    x1, y1, x2, y2 = item["bbox_xyxy"]
                    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    fh.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        rows.extend(train_rows)
    (out / "pseudo" / "dataset.yaml").write_text(
        f"path: {out / 'pseudo'}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: traffic light\n",
        encoding="utf-8")
    (out / "meta" / "pseudo_labels.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "meta" / "audit.json").write_text(json.dumps({
        "videos": sorted({r["video"] for r in rows}),
        "frames_indexed": len(index),
        "frames_processed": len({r["image"] for r in rows}),
        "labels": len(rows),
        "review_counts": dict(counts),
        "color_counts": dict(color_counts),
        "size_counts": dict(size_counts),
        "source": "competition_video_pseudo",
        "human_gt_separate": True,
    }, indent=2), encoding="utf-8")
    print("labels", len(rows), "review", counts, "colors", color_counts, "sizes", size_counts)


if __name__ == "__main__":
    main()

