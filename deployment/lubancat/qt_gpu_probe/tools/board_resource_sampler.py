#!/usr/bin/env python3
"""Sample RK3566 CPU threads and devfreq/thermal counters during a live run."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import psutil


def read_text(path: Path):
    try:
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None


def read_int(path: Path):
    raw = read_text(path)
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def devfreq(path: Path):
    load = read_text(path / "load")
    load_percent = None
    if load:
        try:
            load_percent = float(load.split("@", 1)[0])
        except ValueError:
            pass
    return {
        "load_percent": load_percent,
        "frequency_hz": read_int(path / "cur_freq"),
    }


def role_for(process):
    try:
        command = " ".join(process.cmdline()).lower()
        name = process.name().lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    if "ai_bridge.py" in command:
        return "sidecar"
    if "spawn_main" in command:
        return "ocr"
    if process.name() == "qt_gpu_probe":
        return "qt"
    if "pango-video-capture" in command or "pango_video_capture" in command:
        return "pcie_capture"
    if "rknn_server" in command or name == "rknn_server":
        return "rknn_server"
    return None


def cpu_counters():
    counters = {}
    try:
        for line in Path("/proc/stat").read_text(encoding="ascii").splitlines():
            fields = line.split()
            if not fields or not fields[0].startswith("cpu") or fields[0] == "cpu":
                continue
            values = [int(value) for value in fields[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            counters[fields[0]] = (sum(values), idle)
    except (OSError, ValueError):
        pass
    return counters


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def summarize(values):
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.duration <= 0 or args.interval <= 0:
        parser.error("duration and interval must be positive")

    output = Path(args.output)
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")

    targets = {}
    for process in psutil.process_iter():
        role = role_for(process)
        if role is not None:
            targets.setdefault(role, {})[process.pid] = process
    for process in targets.get("sidecar", {}).values():
        try:
            for child in process.children(recursive=True):
                role = role_for(child)
                if role is not None:
                    targets.setdefault(role, {})[child.pid] = child
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    previous_process = {}
    previous_threads = {}
    previous_cpu = cpu_counters()
    samples = []
    started = time.monotonic()
    started_monotonic_ns = time.monotonic_ns()
    end = started + args.duration
    while time.monotonic() < end:
        target_time = started + (len(samples) + 1) * args.interval
        time.sleep(max(0.0, target_time - time.monotonic()))
        cpu_now = cpu_counters()
        per_core = []
        for name in sorted(cpu_now, key=lambda value: int(value[3:])):
            current_total, current_idle = cpu_now[name]
            previous_total, previous_idle = previous_cpu.get(name, (current_total, current_idle))
            total_delta = current_total - previous_total
            idle_delta = current_idle - previous_idle
            per_core.append(100.0 * max(0, total_delta - idle_delta) / total_delta if total_delta > 0 else 0.0)
        sampled_at = time.monotonic()
        interval = args.interval if not samples else sampled_at - samples[-1]["_monotonic_s"]
        roles = {}
        thread_values = {}
        current_process = {}
        current_threads = {}

        for role, processes in targets.items():
            for pid, process in list(processes.items()):
                try:
                    cpu_time = sum(process.cpu_times()[:2])
                    rss = process.memory_info().rss
                    threads = process.threads()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    processes.pop(pid, None)
                    continue
                current_process[pid] = cpu_time
                row = roles.setdefault(role, {"pids": [], "cpu_percent": [], "rss_mb": [], "threads": 0})
                row["pids"].append(pid)
                row["rss_mb"].append(rss / (1024 * 1024))
                row["threads"] += len(threads)
                if pid in previous_process:
                    row["cpu_percent"].append(max(0.0, (cpu_time - previous_process[pid]) * 100.0 / interval))
                for thread in threads:
                    key = (pid, thread.id)
                    thread_time = thread.user_time + thread.system_time
                    current_threads[key] = thread_time
                    if key in previous_threads:
                        thread_percent = max(0.0, (thread_time - previous_threads[key]) * 100.0 / interval)
                        comm = read_text(Path(f"/proc/{pid}/task/{thread.id}/comm")) or str(thread.id)
                        thread_values[f"{role}:{pid}:{thread.id}:{comm}"] = thread_percent

        npu = devfreq(Path("/sys/class/devfreq/fde40000.npu"))
        gpu = devfreq(Path("/sys/devices/platform/fde60000.gpu/devfreq/fde60000.gpu"))
        dmc = devfreq(Path("/sys/class/devfreq/dmc"))
        cpu_frequency = {
            path.parent.parent.name: read_int(path)
            for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")
        }
        temperatures = {}
        for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
            name = read_text(zone / "type")
            value = read_int(zone / "temp")
            if name and value is not None:
                temperatures[name] = value / 1000.0

        samples.append({
            "elapsed_s": sampled_at - started,
            "monotonic_ns": time.monotonic_ns(),
            "_monotonic_s": sampled_at,
            "cpu_per_core_percent": per_core,
            "cpu_frequency_khz": cpu_frequency,
            "roles": {
                name: {
                    "pids": row["pids"],
                    "cpu_percent": statistics.fmean(row["cpu_percent"]) if row["cpu_percent"] else None,
                    "rss_mb": sum(row["rss_mb"]),
                    "thread_count": row["threads"],
                }
                for name, row in roles.items()
            },
            "top_thread_cpu_percent": sorted(
                ({"thread": name, "cpu_percent": value} for name, value in thread_values.items()),
                key=lambda item: item["cpu_percent"], reverse=True,
            )[:16],
            "npu": npu,
            "gpu": gpu,
            "dmc": dmc,
            "temperatures_c": temperatures,
        })
        previous_process = current_process
        previous_threads = current_threads
        previous_cpu = cpu_now

    per_core_count = len(samples[0]["cpu_per_core_percent"]) if samples else 0
    summaries = {
        "cpu_per_core_percent": [
            summarize([row["cpu_per_core_percent"][core] for row in samples])
            for core in range(per_core_count)
        ],
        "roles": {},
        "cpu_frequency_khz": {},
    }
    for cpu in sorted({name for row in samples for name in row["cpu_frequency_khz"]}):
        summaries["cpu_frequency_khz"][cpu] = summarize([
            row["cpu_frequency_khz"][cpu]
            for row in samples
            if row["cpu_frequency_khz"].get(cpu) is not None
        ])
    for role in sorted({name for row in samples for name in row["roles"]}):
        rows = [row["roles"][role] for row in samples if role in row["roles"]]
        summaries["roles"][role] = {
            "cpu_percent": summarize([row["cpu_percent"] for row in rows if row["cpu_percent"] is not None]),
            "rss_mb": summarize([row["rss_mb"] for row in rows]),
            "thread_count": summarize([float(row["thread_count"]) for row in rows]),
        }
    for key in ("npu", "gpu", "dmc"):
        summaries[key] = {
            metric: summarize([row[key][metric] for row in samples if row[key][metric] is not None])
            for metric in ("load_percent", "frequency_hz")
        }
    summaries["temperatures_c"] = {
        name: summarize([row["temperatures_c"][name] for row in samples if name in row["temperatures_c"]])
        for name in sorted({name for row in samples for name in row["temperatures_c"]})
    }
    summaries["top_threads"] = sorted(
        (item for row in samples for item in row["top_thread_cpu_percent"]),
        key=lambda item: item["cpu_percent"], reverse=True,
    )[:32]
    for row in samples:
        row.pop("_monotonic_s", None)
    report = {
        "duration_s": time.monotonic() - started,
        "started_monotonic_ns": started_monotonic_ns,
        "finished_monotonic_ns": time.monotonic_ns(),
        "sample_count": len(samples),
        "memory_bandwidth_note": "DMC load is the driver-reported busy percentage; this kernel exposes no bytes-per-second counter.",
        "summaries": summaries,
        "samples": samples,
    }
    output.write_text(json.dumps(report, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("duration_s", "sample_count", "summaries")}, indent=2))


if __name__ == "__main__":
    main()
