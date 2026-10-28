"""Evaluate a small, human-reviewed signal audit against benchmark CSV rows.

The labels are intentionally a narrow audit of the supplied competition clip,
not a dataset-wide accuracy claim.  They keep vehicle signals and pedestrian
signals separate so a red pedestrian light cannot count as an ego red light.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _metric(tp: int, fp: int, fn: int, negative_frames: int) -> dict[str, float | int | None]:
    # Positive-only labels can establish recall, but cannot establish precision.
    # UNKNOWN means the reviewer did not assert a class, not a negative example.
    precision = tp / max(1, tp + fp) if negative_frames else None
    recall = tp / max(1, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(1e-9, precision + recall), 4) if precision is not None else None,
        "negative_frames_reviewed": negative_frames,
    }


def evaluate(rows: dict[int, dict[str, str]], labels: dict, *, vehicle_prediction_key: str = "traffic_light_state", pedestrian_prediction_key: str = "traffic_light_pedestrian_state") -> dict:
    ranges = labels["ranges"]
    expected: dict[int, dict[str, str]] = {}
    for item in ranges:
        for frame_id in range(int(item["start_frame"]), int(item["end_frame"]) + 1):
            if frame_id in expected:
                raise ValueError(f"overlapping label ranges at frame {frame_id}")
            expected[frame_id] = item
    result = {"labelled_frame_range_count": len(expected), "metrics_scope": "COLOUR_STATE_FRAME_RECALL_NOT_OBJECT_DETECTION_AP"}
    for name, target, label_key, prediction_key, fallback in (
        ("vehicle_green", "GREEN", "vehicle_state", vehicle_prediction_key, "state"),
        ("pedestrian_red", "RED", "pedestrian_state", pedestrian_prediction_key, "ped"),
    ):
        tp = fp = fn = negatives = positives = excluded = missing = 0
        for frame_id, item in expected.items():
            wanted = str(item.get(label_key, "UNREVIEWED")).upper()
            if wanted in {"UNKNOWN", "UNREVIEWED"}:
                excluded += 1
                continue
            row = rows.get(frame_id)
            if row is None:
                missing += 1
            actual = str((row or {}).get(prediction_key, (row or {}).get(fallback, "UNKNOWN"))).upper()
            if wanted == target:
                positives += 1
                tp += actual == target
                fn += actual != target
            else:
                # RED/YELLOW/GREEN or explicit ABSENT are reviewed alternatives.
                if wanted not in {"RED", "YELLOW", "GREEN", "ABSENT"}:
                    raise ValueError(f"unsupported audit label: {wanted}")
                negatives += 1
                fp += actual == target
        result[name] = {
            **_metric(tp, fp, fn, negatives),
            "positive_frames_reviewed": positives,
            "excluded_ambiguous_or_unreviewed_frames": excluded,
            "missing_prediction_frames": missing,
        }
    reviewed_negatives = any(result[name]["negative_frames_reviewed"] > 0 for name in ("vehicle_green", "pedestrian_red"))
    result["interpretation"] = (
        "Narrow visual clip audit. UNKNOWN/UNREVIEWED labels excluded; precision and F1 use only explicitly reviewed negative/alternative frames. Not dataset-wide accuracy or ego-lane association."
        if reviewed_negatives
        else "Narrow visual clip audit. UNKNOWN/UNREVIEWED labels excluded; precision and F1 unavailable without reviewed negatives. Not dataset-wide accuracy or ego-lane association."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vehicle-prediction-key", default="traffic_light_state")
    parser.add_argument("--pedestrian-prediction-key", default="traffic_light_pedestrian_state")
    args = parser.parse_args()

    import csv

    loaded_rows = list(csv.DictReader(args.frames.open(encoding="utf-8")))
    frame_key = "frame_id" if loaded_rows and "frame_id" in loaded_rows[0] else "frame"
    rows = {int(row[frame_key]): row for row in loaded_rows}
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    result = {
        "frames_csv": str(args.frames.resolve()),
        "labels": str(args.labels.resolve()),
        "label_provenance": labels.get("provenance"),
        **evaluate(rows, labels, vehicle_prediction_key=args.vehicle_prediction_key, pedestrian_prediction_key=args.pedestrian_prediction_key),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
