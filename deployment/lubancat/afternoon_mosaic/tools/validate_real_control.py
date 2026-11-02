"""Explicitly authorized real brake command; FPGA feedback is not mechanical proof."""
import argparse
import json
import socket
import time
import uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    report = dict(kind="REAL_PCIE_EXPLICIT_CONTROL_PROBE", simulated=False,
                  physical_actuator_verified=False, samples=[])
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(1)
            for request in [dict(type="action", action="status"), dict(type="action", action="brake",
                    source="user_authorized_competition_probe", request_id=uuid.uuid4().hex)]:
                sock.sendto(json.dumps(request).encode(), ("127.0.0.1",38262))
                reply = json.loads(sock.recv(4096))
                report["samples"].append(dict(timestamp_ns=time.monotonic_ns(),request=request,reply=reply))
            for _ in range(16):
                report["samples"].append(dict(timestamp_ns=time.monotonic_ns(),
                    vehicle=json.loads(Path("/dev/shm/adas_vehicle_status.json").read_text())))
                time.sleep(.1)
        report["accepted"] = report["samples"][1]["reply"].get("fpga_accepted",False)
        report["fpga_active_seen"] = any(r.get("vehicle",{}).get("current_action")=="brake" for r in report["samples"])
        report["fpga_brake_allowed_seen"] = any(r.get("vehicle",{}).get("brake_allow") for r in report["samples"])
        report["ended"] = report["samples"][-1].get("vehicle",{}).get("current_action")=="none"
        report["passed"] = all(report.get(k) for k in ("accepted","fpga_active_seen","fpga_brake_allowed_seen","ended"))
    except Exception as exc:
        report.update(passed=False,error=repr(exc))
        raise
    finally:
        (ROOT/"evidence"/(args.label+"_manual_control.json")).write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps({k:v for k,v in report.items() if k!="samples"}),flush=True)
if __name__ == "__main__": main()
