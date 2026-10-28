"""Exercise HUD v1 on the PC reference register model; no hardware access."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from competition_io.hud_client import HudClient, OutstandingTransaction
from competition_io.hud_simulator import HudSimulator
from competition_io.hud_v1 import Register as R, VECTORS


def run_demo(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    bus = HudSimulator()
    sender = HudClient(bus, clock=bus.clock, sleep=bus.sleep)
    results = []
    for command, expected in VECTORS:
        result = sender.submit_full_state(command.owner, command.state, command.blink)
        assert result.word == expected and result.applied
        results.append(asdict(result))

    # Trace the visible state, including the complete OFF half of each blink.
    bus = HudSimulator()
    sender = HudClient(bus, clock=bus.clock, sleep=bus.sleep)
    sender.submit_full_state(True, 0x103, 1)
    trace = []
    for frame in range(120):
        bus.dehaze_input = 40 <= frame < 80
        if frame:
            bus.advance_frames()
        trace.append({"frame": frame, "time_ms": round(bus.clock()*1000, 3),
                      "dehaze": int(bus.covered), "rk_state": bus.read32(R.RK_STATE),
                      "effective": bus.effective, "left": int(bool(bus.effective & 2)),
                      "right": int(bool(bus.effective & 4)), "hazard": int(bool(bus.effective & 8))})
    assert [row["left"] for row in trace[:40]] == [1]*20 + [0]*20
    assert [row["left"] for row in trace[40:80]] == [1]*20 + [0]*20
    assert [row["left"] for row in trace[80:120]] == [1]*20 + [0]*20
    with (output / "hud_frame_trace.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)

    bus.pixel_running = False
    timed_out = sender.submit_full_state(True, 0x105, 2)
    assert timed_out.phase == "UNKNOWN" and timed_out.accepted
    writes_before = len(bus.writes)
    try:
        sender.submit_full_state(True, 0x101)
    except OutstandingTransaction:
        pass
    else:
        raise AssertionError("A timeout incorrectly allowed another command")
    assert len(bus.writes) == writes_before
    bus.pixel_running = True
    bus.advance_frames()
    resumed = sender.query_pending()
    assert resumed and resumed.applied

    bus.dehaze_input = True
    covered = sender.submit_full_state(True, 0)
    assert covered.applied and covered.covered and covered.effective_state & 1
    bus.dehaze_input = False
    bus.advance_frames()
    assert bus.effective == 0
    released = sender.release()
    assert released.applied and bus.effective == 0x101

    bus.pixel_running = False
    sender.submit_full_state(True, 0x103, 1)
    bus.reset()
    sender.notify_device_reset()
    bus.pixel_running = True
    bus.advance_frames()
    assert bus.applied is None and bus.effective == 0x101
    report = {"evidence": "PC_REGISTER_REFERENCE_MODEL_NOT_RTL_OR_BOARD", "passed": True,
              "vectors": results, "trace_frames": len(trace), "timeout": asdict(timed_out),
              "resume": asdict(resumed), "covered": asdict(covered), "release": asdict(released),
              "reset_cancelled_pending": True, "timeout_blocked_second_write": True,
              "limitations": ["No PCIe transactions, CDC, FPGA synthesis, STA or physical HDMI validation"]}
    (output / "hud_demo.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_demo(arguments.output)
    print(json.dumps({"passed": result["passed"], "evidence": result["evidence"],
                      "output": str(arguments.output.resolve())}, ensure_ascii=False))
