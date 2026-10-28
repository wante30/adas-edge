#!/usr/bin/env python3
"""Summarize one monotonic-time window from the live RK3566 audit logs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import re
import statistics
import subprocess
import time


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def distribution(values):
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def stage_stats(rows, prefix="", fields=()):
    container = prefix.rstrip(".")
    return {
        field: distribution([
            float(((row.get(container) or {}) if container else row).get(field))
            for row in rows
            if isinstance(((row.get(container) or {}) if container else row).get(field), (int, float))
        ])
        for field in fields
    }


def resource_summary(resource):
    result = dict(resource["summaries"])
    frequencies = {}
    for sample in resource.get("samples", []):
        for cpu, value in sample.get("cpu_frequency_khz", {}).items():
            if value is not None:
                frequencies.setdefault(cpu, []).append(float(value))
    result["cpu_frequency_khz"] = {
        cpu: distribution(values) for cpu, values in sorted(frequencies.items())
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qt-jsonl", required=True)
    parser.add_argument("--ai-jsonl", required=True)
    parser.add_argument("--resource-json", required=True)
    parser.add_argument("--qt-anchor-monotonic-ns", type=int, required=True)
    parser.add_argument("--qt-anchor-elapsed-ms", type=float, required=True)
    parser.add_argument("--journal-unit", default="adas-ai-bridge-live.service")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    resource = json.load(open(args.resource_json, "r", encoding="utf-8"))
    if "started_monotonic_ns" in resource:
        start_ns = int(resource["started_monotonic_ns"])
        end_ns = int(resource["finished_monotonic_ns"])
    else:
        wall_mono_offset = time.time_ns() - time.monotonic_ns()
        end_ns = int(__import__("os").stat(args.resource_json).st_mtime_ns - wall_mono_offset)
        start_ns = int(end_ns - float(resource["duration_s"]) * 1e9)
    duration_s = (end_ns - start_ns) / 1e9

    qt_process_start_ns = int(args.qt_anchor_monotonic_ns - args.qt_anchor_elapsed_ms * 1e6)
    qt_start_ms = (start_ns - qt_process_start_ns) / 1e6
    qt_end_ms = (end_ns - qt_process_start_ns) / 1e6
    qt_rows = [row for row in read_jsonl(args.qt_jsonl)
               if qt_start_ms <= float(row.get("process_elapsed_ms", -1)) <= qt_end_ms]
    if len(qt_rows) < 2:
        raise SystemExit(f"only {len(qt_rows)} Qt metric samples fall inside {args.label}")
    qt_elapsed_s = (qt_rows[-1]["process_elapsed_ms"] - qt_rows[0]["process_elapsed_ms"]) / 1000.0
    qt_delta = {}
    for counter in ("uploaded_frames", "produced_frames", "ai_requests", "ai_results"):
        qt_delta[counter] = qt_rows[-1][counter] - qt_rows[0][counter]

    qt_rates = {
        name: distribution([
            float(row[name])
            for row in qt_rows
            if isinstance(row.get(name), (int, float))
        ])
        for name in ("texture_upload_fps", "scene_graph_fps", "source_fps")
    }
    qt_summary = {
        "elapsed_s": qt_elapsed_s,
        "sample_count": len(qt_rows),
        "unique_uploaded_frame_fps": qt_delta["uploaded_frames"] / qt_elapsed_s,
        "produced_source_frame_fps": qt_delta["produced_frames"] / qt_elapsed_s,
        "ai_requests_fps": qt_delta["ai_requests"] / qt_elapsed_s,
        "ai_results_fps": qt_delta["ai_results"] / qt_elapsed_s,
        "windowed_rates": qt_rates,
        "result_age_ms": distribution([
            float(row["ai_result_age_ms"])
            for row in qt_rows
            if isinstance(row.get("ai_result_age_ms"), (int, float)) and row["ai_results"] > 0
        ]),
        "gui_update_ms": distribution([
            float(row["ai_gui_update_ms"])
            for row in qt_rows
            if isinstance(row.get("ai_gui_update_ms"), (int, float)) and row["ai_results"] > 0
        ]),
    }

    ai_rows = [row for row in read_jsonl(args.ai_jsonl)
               if start_ns <= int(row.get("result_publish_timestamp_ns", 0)) <= end_ns]
    main_rows = [row for row in ai_rows if row.get("main_ran") and isinstance(row.get("main_timing"), dict)]
    specialist_rows = [row for row in ai_rows if row.get("specialist_ran") and isinstance(row.get("specialist_timing"), dict)]
    inference_latency = [
        (int(row["result_publish_timestamp_ns"]) - int(row["capture_timestamp_ns"])) / 1e6
        for row in ai_rows
        if int(row.get("capture_timestamp_ns", 0)) > 0
    ]
    ai_stage_fields = (
        "shared_frame_read_ms", "shared_frame_validate_ms", "decode_ms", "pipeline_ms",
        "navigation_submit_ms", "box_mapping_ms", "status_build_ms", "dynamic_hmi_ms",
        "packet_encode_ms", "result_publish_ms",
    )
    detector_fields = ("preprocess_ms", "rknn_call_ms", "decode_nms_ms", "total_ms")
    ai_summary = {
        "result_packet_fps": len(ai_rows) / duration_s,
        "yolo_completed_fps": len(main_rows) / duration_s,
        "yolo_completed_count": len(main_rows),
        "specialist_completed_fps": len(specialist_rows) / duration_s,
        "specialist_completed_count": len(specialist_rows),
        "result_age_ms_at_publish": distribution(inference_latency),
        "stages_ms": stage_stats(ai_rows, fields=ai_stage_fields),
        "yolo_ms": stage_stats(main_rows, prefix="main_timing.", fields=detector_fields),
        "specialist_ms": stage_stats(specialist_rows, prefix="specialist_timing.", fields=detector_fields),
        "yolo_preprocess_backends": dict(Counter(
            row["main_timing"].get("preprocess_backend", "unknown") for row in main_rows
        )),
    }

    journal = subprocess.run(
        ["journalctl", "-b", "-u", args.journal_unit, "-o", "json", "--no-pager"],
        check=True, capture_output=True, text=True,
    )
    ocr_rows = []
    for line in journal.stdout.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        timestamp_ns = int(entry.get("__MONOTONIC_TIMESTAMP", 0)) * 1000
        message = str(entry.get("MESSAGE", ""))
        if not (start_ns <= timestamp_ns <= end_ns) or not message.startswith("NAV_OCR_RESULT"):
            continue
        match = re.search(r"\bengine_ms=([0-9.]+)", message)
        if match:
            ocr_rows.append({"engine_ms": float(match.group(1)), "message": message})
    ocr_engine = [row["engine_ms"] for row in ocr_rows if row["engine_ms"] > 5.0]
    ai_summary["ocr"] = {
        "reported_result_count": len(ocr_rows),
        "compute_call_count": len(ocr_engine),
        "compute_fps": len(ocr_engine) / duration_s,
        "engine_ms": distribution(ocr_engine),
    }

    report = {
        "label": args.label,
        "duration_s": duration_s,
        "monotonic_start_ns": start_ns,
        "monotonic_end_ns": end_ns,
        "qt_elapsed_window_ms": [qt_start_ms, qt_end_ms],
        "qt": qt_summary,
        "ai": ai_summary,
        "resources": resource_summary(resource),
        "memory_bandwidth_note": resource.get("memory_bandwidth_note"),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
