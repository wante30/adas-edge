#!/usr/bin/env python3
"""Join sidecar and Qt JSONL timing records and report latency percentiles."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read_jsonl(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                frame = int(row["frame_id"])
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            rows[frame] = row
    return rows


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[pos]


def summary(values: list[float]) -> dict:
    return {
        "n": len(values),
        "p50": pct(values, 0.50),
        "p95": pct(values, 0.95),
        "mean": statistics.fmean(values) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai-log", type=Path, required=True)
    parser.add_argument("--qt-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ai = read_jsonl(args.ai_log)
    qt = read_jsonl(args.qt_log)
    common = sorted(set(ai) & set(qt))
    rows = []
    for frame in common:
        a, q = ai[frame], qt[frame]
        rows.append({
            "frame_id": frame,
            "capture_to_ai_start_ms": (a["infer_start_ns"] - a["capture_timestamp_ns"]) / 1e6,
            "ai_receive_to_start_ms": (a["infer_start_ns"] - a["ai_receive_timestamp_ns"]) / 1e6,
            "infer_ms": (a["infer_end_ns"] - a["infer_start_ns"]) / 1e6,
            "publish_to_qt_receive_ms": (q["qt_receive_ns"] - a["result_publish_timestamp_ns"]) / 1e6,
            "qt_apply_after_publish_ms": (q.get("qt_apply_ns", q["qt_receive_ns"]) -
                                           a["result_publish_timestamp_ns"]) / 1e6,
            "result_age_at_qt_ms": (q["qt_receive_ns"] - a["capture_timestamp_ns"]) / 1e6,
            "result_age_at_apply_ms": (q.get("qt_apply_ns", q["qt_receive_ns"]) -
                                       a["capture_timestamp_ns"]) / 1e6,
            "frame_lag_at_receive": int(q.get("display_frame_id", 0)) - frame,
            "boxes": a.get("boxes"),
        })
    publishes = sorted(a["result_publish_timestamp_ns"] for a in ai.values())
    intervals = [(b - a) / 1e6 for a, b in zip(publishes, publishes[1:])]
    report = {
        "frames_joined": len(rows),
        "ai_rows": len(ai),
        "qt_rows": len(qt),
        "capture_to_ai_start_ms": summary([r["capture_to_ai_start_ms"] for r in rows]),
        "ai_receive_to_start_ms": summary([r["ai_receive_to_start_ms"] for r in rows]),
        "infer_ms": summary([r["infer_ms"] for r in rows]),
        "publish_to_qt_receive_ms": summary([r["publish_to_qt_receive_ms"] for r in rows]),
        "result_age_at_qt_ms": summary([r["result_age_at_qt_ms"] for r in rows]),
        "result_age_at_apply_ms": summary([r["result_age_at_apply_ms"] for r in rows]),
        "frame_lag_at_receive": summary([float(r["frame_lag_at_receive"]) for r in rows]),
        "result_interval_ms": summary(intervals),
        "result_gaps_over_500ms": sum(1 for value in intervals if value > 500.0),
        "result_gaps_over_1000ms": sum(1 for value in intervals if value > 1000.0),
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
