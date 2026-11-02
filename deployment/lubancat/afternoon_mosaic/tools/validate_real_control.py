"""Bounded MANUAL competition-bench probe; never represents an AI decision."""
import argparse
import json
import socket
import time
import uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--count", type=int, choices=range(1, 4), default=1)
    args = parser.parse_args()
    report = dict(kind="REAL_PCIE_MANUAL_AUTHORIZED_CONTROL_PROBE", simulated=False,
                  ai_triggered=False, physical_actuator_verified=False, probes=[])
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(1)
            def exchange(request):
                sock.sendto(json.dumps(request).encode(), ("127.0.0.1", 38262))
                reply = json.loads(sock.recv(8192))
                if reply.get("request_id") != request["request_id"]:
                    raise RuntimeError("ACK request_id mismatch")
                return reply
            preflight = dict(type="action", action="status", request_id=uuid.uuid4().hex)
            report["preflight"] = exchange(preflight)
            if not report["preflight"].get("ok"):
                raise RuntimeError("FPGA action status unavailable")
            for _ in range(args.count):
                request = dict(type="action", action="brake", source="user_authorized_competition_probe",
                               request_id=uuid.uuid4().hex)
                row = dict(timestamp_ns=time.monotonic_ns(), request=request,
                           reply=exchange(request), samples=[])
                report["probes"].append(row)
                sequence = row["reply"].get("sequence")
                for _ in range(32):
                    row["samples"].append(dict(timestamp_ns=time.monotonic_ns(),
                        vehicle=json.loads(Path("/dev/shm/adas_vehicle_status.json").read_text())))
                    time.sleep(.05)
                row["accepted"] = row["reply"].get("fpga_accepted") is True
                row["ack_active"] = row["reply"].get("fpga_action_active") is True
                row["sequence"] = sequence
                row["correlated_active"] = sequence is not None and any(
                    x["vehicle"].get("last_sequence") == sequence and
                    x["vehicle"].get("current_action") == "brake" and
                    x["vehicle"].get("brake_allow") and
                    x["vehicle"].get("online") and not x["vehicle"].get("simulated", False)
                    for x in row["samples"])
                row["ended"] = row["samples"][-1]["vehicle"].get("current_action") == "none"
                row["passed"] = all(row[k] for k in ("accepted", "ack_active", "correlated_active", "ended"))
                if not row["passed"]:
                    break
        report["passed"] = len(report["probes"]) == args.count and all(x["passed"] for x in report["probes"])
    except Exception as exc:
        report.update(passed=False, error=repr(exc))
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / (args.label + "_manual_control.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "probes"}), flush=True)
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
