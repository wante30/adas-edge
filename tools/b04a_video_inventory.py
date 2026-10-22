#!/usr/bin/env python3
"""B-04A: read-only, permission-gated inventory of explicitly listed source videos.

No video frames are exported. This tool proposes timestamps only; proposals are
NOT training data until images are extracted, independently reviewed, and fully
annotated. Never pass hidden competition test video as training material.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ("video_path", "source_id", "source_kind", "permission", "split", "license_note")
SPLITS = {"train", "val", "sealed-test", "unassigned"}
PERMISSIONS = {"allowed", "unknown", "forbidden"}
KINDS = {"team_recorded", "competition_development", "official_test", "third_party"}
SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def sample_times(duration_s: float, gap_s: float, max_per_video: int) -> list[float]:
    if not math.isfinite(duration_s) or duration_s <= 0:
        return []
    if not math.isfinite(gap_s) or gap_s <= 0 or max_per_video < 1:
        raise ValueError("invalid sampling interval or limit")
    # Avoid the very first/last video frame and near-duplicate neighboring frames.
    points = []
    current = min(gap_s / 2, duration_s / 2)
    while current < duration_s - 0.1 and len(points) < max_per_video:
        points.append(round(current, 3))
        current += gap_s
    return points


def eligible(row: dict[str, str]) -> tuple[bool, str]:
    if row["source_kind"] == "official_test":
        return False, "official_test_never_train"
    if row["permission"] != "allowed" or not row["license_note"].strip():
        return False, "training_rights_unverified"
    if row["split"] not in {"train", "val"}:
        return False, "not_training_split"
    return True, "timestamp_proposals_only"


def inspect_video(path: Path) -> tuple[float | None, str]:
    try:
        import cv2
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return None, "video_cannot_open"
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0 or not math.isfinite(frames) or frames <= 0:
                return None, "video_duration_unknown"
            return frames / fps, "ok"
        finally:
            capture.release()
    except ImportError:
        return None, "opencv_not_installed"


def inventory(registry: Path, gap_s: float = 5.0, max_total: int = 800, max_per_video: int = 150):
    if max_total < 1 or max_per_video < 1:
        raise ValueError("sample limits must be positive")
    with registry.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("missing registry columns: " + ", ".join(sorted(missing)))
        rows = [{key: (row.get(key) or "").strip() for key in FIELDS} for row in reader]
    report = {"status": "PARTIAL", "videos": [], "errors": [], "warnings": [],
              "summary": {}, "training_ready": False}
    groups: dict[str, set[str]] = defaultdict(set)
    choices: list[list[dict[str, object]]] = []
    for i, row in enumerate(rows, start=2):
        path = Path(row["video_path"])
        errors = []
        for key in ("video_path", "source_id", "source_kind", "permission", "split"):
            if not row[key]:
                errors.append(f"missing_{key}")
        if row["source_kind"] not in KINDS:
            errors.append("invalid_source_kind")
        if row["permission"] not in PERMISSIONS:
            errors.append("invalid_permission")
        if row["split"] not in SPLITS:
            errors.append("invalid_split")
        if path.suffix.lower() not in SUFFIXES:
            errors.append("unsupported_video_extension")
        if not path.is_file():
            errors.append("video_not_found")
        if row["source_id"] and row["split"] in SPLITS:
            groups[row["source_id"]].add(row["split"])
        ok, reason = eligible(row)
        duration, probe_status = (inspect_video(path) if not errors else (None, "skipped"))
        if ok and duration is None:
            errors.append(probe_status)
        if errors:
            ok = False
        report["videos"].append({"registry_line": i, "source_id": row["source_id"],
                                 "source_kind": row["source_kind"], "permission": row["permission"],
                                 "split": row["split"], "video_path": str(path),
                                 "duration_seconds": round(duration, 3) if duration is not None else None,
                                 "candidate_status": reason if ok else reason,
                                 "probe_status": probe_status, "errors": errors})
        report["errors"].extend(f"row {i}: {e}" for e in errors)
        if not ok:
            report["warnings"].append(f"row {i}: excluded ({reason}; row errors={len(errors)})")
            choices.append([])
        else:
            choices.append([{"source_id": row["source_id"], "video_path": str(path),
                             "split": row["split"], "timestamp_s": t,
                             "annotation_status": "NOT_ANNOTATED",
                             "training_ready": "false"}
                            for t in sample_times(duration, gap_s, max_per_video)])
    for source_id, splits in groups.items():
        if len(splits) > 1:
            report["errors"].append(
                f"video/source leakage: {source_id}: {','.join(sorted(splits))}")
    proposals = []
    # Round-robin across videos rather than letting the first video use the whole budget.
    for k in range(max((len(c) for c in choices), default=0)):
        for candidates in choices:
            if k < len(candidates) and len(proposals) < max_total:
                proposals.append(candidates[k])
    if report["errors"]:
        # Never provide actionable frame proposals when any source or split is invalid.
        proposals = []
    counts = Counter(item["split"] for item in proposals)
    report["summary"] = {"registry_videos": len(rows),
                         "timestamp_proposals": len(proposals),
                         "proposed_splits": dict(counts),
                         "error_count": len(report["errors"]),
                         "video_permission_review_required": True,
                         "full_10_class_annotations_verified": False}
    report["status"] = "FAIL" if report["errors"] else "PARTIAL"
    report["warnings"].append(
        "Proposals contain no images or labels. Do not train until source permissions, "
        "video-level separation and complete per-class annotations are independently verified.")
    return report, proposals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gap-seconds", type=float, default=5.0)
    parser.add_argument("--max-total", type=int, default=800)
    parser.add_argument("--max-per-video", type=int, default=150)
    args = parser.parse_args()
    result, proposals = inventory(args.registry, args.gap_seconds,
                                  args.max_total, args.max_per_video)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "B04A_inventory.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "B04A_frame_proposals.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=(
            "source_id", "video_path", "split", "timestamp_s",
            "annotation_status", "training_ready"))
        writer.writeheader()
        writer.writerows(proposals)
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(f"status={result['status']} ; NO video frames were extracted")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
