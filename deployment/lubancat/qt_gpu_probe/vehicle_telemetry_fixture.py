#!/usr/bin/env python3
"""Offline LVT1 fixture publisher for Qt HMI development.

This writes the same schema as vehicle_telemetry_v1.py so the Qt
VehicleTelemetryStatus object can be exercised without FPGA/driver/PCIe.
It is deliberately marked simulated=true; real reader output must not be
labelled simulated.

Usage:
  python3 vehicle_telemetry_fixture.py --scenario lane-left-solid-right-dashed
  python3 vehicle_telemetry_fixture.py --cycle --period 2.5 --duration 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

SCHEMA = "pango.rk3566.vehicle_status.v1"

BASE = {
    "schema": SCHEMA,
    "online": True,
    "simulated": True,
    "generation": 1,
    "hsst_sample_seq": 42,
    "hsst_valid": True,
    "hsst_stale": False,
    "hsst_fresh": True,
    "hsst_owns_both_sides": True,
    "rk_scene_fresh": True,
    "lane_source": "hsst",
    "left_line_count": 2,
    "right_line_count": 1,
    "left_nearest_type": "solid",
    "right_nearest_type": "dashed",
    "current_action": "none",
    "last_action": "none",
    "last_sequence": 1,
    "active": False,
    "timebase_ready": True,
    "accept_count": 1,
    "decision": "idle",
    "decision_action_active": False,
    "reason_code": 0,
    "reason": "none",
    "sensor_stale": False,
    "throttle_allow": False,
    "brake_allow": False,
    "lane_left_allow": False,
    "lane_right_allow": False,
    "throttle_deny": False,
    "brake_deny": False,
    "lane_left_deny": False,
    "lane_right_deny": False,
    "action_sequence_low8": 1,
    "rk_scene_sequence_low8": 1,
    "action_hold_ms": 1001,
    "hsst_stale_ms": 500,
}


def scenario(name: str) -> dict:
    d = dict(BASE)
    d["scenario"] = name
    if name == "lane-left-solid-right-dashed":
        return d
    if name == "lane-left-dashed-right-solid":
        d.update(left_line_count=1, right_line_count=2,
                 left_nearest_type="dashed", right_nearest_type="solid")
        return d
    if name == "lane-count-change":
        d.update(left_line_count=3, right_line_count=2,
                 left_nearest_type="solid", right_nearest_type="dashed")
        return d
    if name == "left-change-denied":
        d.update(current_action="lane_left", last_action="lane_left", active=True,
                 decision="deny", decision_action_active=True,
                 lane_left_deny=True, reason_code=2, reason="left_solid",
                 left_nearest_type="solid", accept_count=2, last_sequence=2)
        return d
    if name == "right-change-allowed":
        d.update(current_action="lane_right", last_action="lane_right", active=True,
                 decision="allow", decision_action_active=True,
                 lane_right_allow=True, reason_code=0, reason="none",
                 right_nearest_type="dashed", accept_count=3, last_sequence=3)
        return d
    if name == "brake-warning":
        d.update(current_action="brake", last_action="brake", active=True,
                 decision="warning", decision_action_active=True,
                 brake_allow=True, reason_code=8, reason="brake_advisory",
                 accept_count=4, last_sequence=4)
        return d
    if name == "stale":
        d.update(sensor_stale=True, hsst_stale=True, hsst_fresh=False,
                 decision="unknown", decision_action_active=False,
                 current_action="none", active=False,
                 left_nearest_type="unknown", right_nearest_type="unknown",
                 left_line_count=None, right_line_count=None)
        return d
    if name == "offline":
        d.update(online=False, sensor_stale=True, hsst_stale=True, hsst_fresh=False,
                 lane_source="none", decision="unknown", decision_action_active=False,
                 current_action="none", active=False,
                 left_nearest_type="unknown", right_nearest_type="unknown",
                 left_line_count=None, right_line_count=None)
        return d
    raise SystemExit(f"unknown scenario: {name}")


SCENARIOS = [
    "lane-left-solid-right-dashed",
    "lane-left-dashed-right-solid",
    "lane-count-change",
    "left-change-denied",
    "right-change-allowed",
    "brake-warning",
    "stale",
    "offline",
]


def write_atomic(path: Path, status: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=str(path.parent),
                prefix=".lvt1-fixture-", delete=False) as output:
            staged = output.name
            json.dump(status, output, ensure_ascii=False, separators=(",", ":"))
            output.write("\n")
        os.chmod(staged, 0o644)
        os.replace(staged, path)
    finally:
        if staged and os.path.exists(staged):
            os.unlink(staged)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Publish simulated LVT1 status for the Qt HMI")
    ap.add_argument("--scenario", choices=SCENARIOS)
    ap.add_argument("--cycle", action="store_true")
    ap.add_argument("--period", type=float, default=2.5)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--interval-ms", type=int, default=100)
    ap.add_argument("--output", type=Path, default=Path("/dev/shm/adas_vehicle_status.json"))
    args = ap.parse_args(argv)
    if not args.scenario and not args.cycle:
        ap.error("choose --scenario or --cycle")
    if args.interval_ms < 20 or args.interval_ms > 400:
        ap.error("--interval-ms must be 20..400")
    start = time.monotonic()
    index = 0
    while True:
        now = time.monotonic()
        if args.cycle:
            index = int((now - start) / max(0.1, args.period)) % len(SCENARIOS)
            name = SCENARIOS[index]
        else:
            name = args.scenario
        status = scenario(name)
        status["fixture_elapsed_s"] = round(now - start, 3)
        write_atomic(args.output, status)
        print(f"FIXTURE {name} -> {args.output}", file=sys.stderr, flush=True)
        if args.duration > 0 and now - start >= args.duration:
            return 0
        time.sleep(max(0.02, args.interval_ms / 1000.0))


if __name__ == "__main__":
    raise SystemExit(main())
