"""Real BAR1 bridge: preserve deployed HUD protocol, correlate ACKs, publish LVT1."""
import argparse
import json
import socket
import time
from pathlib import Path
from legacy_hud_bridge import Bridge
from vehicle_telemetry_v1 import read_snapshot, write_status_file, TelemetryError


def dispatch(bridge, request):
    if not isinstance(request, dict):
        raise ValueError("JSON object required")
    kind = request.get("type")
    methods = {"status": lambda _: bridge.status(), "action": bridge.action,
               "nav": bridge.nav, "isp": bridge.isp, "protocol": bridge.protocol,
               "mmio": bridge.mmio, "set": bridge.set}
    if kind not in methods:
        raise ValueError("unknown command")
    reply = dict(methods[kind](request))
    reply["control_bridge"] = "ADAS_REAL_PCIE_V1"
    if "request_id" in request:
        reply["request_id"] = request["request_id"]
    if kind == "action" and request.get("action") == "brake":
        # The original bridge checks actual accepted word, error and rejection count.
        # An active FPGA mailbox is not proof of physical actuator execution.
        reply["executed"] = False
        reply["simulated"] = False
        reply["fpga_accepted"] = bool(reply.get("ok") and
            reply.get("accepted") == reply.get("request") and reply.get("request") is not None)
        status = reply.get("status", 0)
        reply["fpga_action_active"] = bool(reply["fpga_accepted"] and status & 2
            and (status >> 5) & 7 == 2 and (status >> 16) & 0xffff == reply.get("sequence"))
    return reply


def publish_vehicle(bridge, output):
    try:
        value = read_snapshot(bridge.rd)
    except (TelemetryError, OSError, ValueError) as exc:
        value = {"schema": "pango.rk3566.vehicle_status.v1", "online": False,
                 "sensor_stale": True, "simulated": False, "error": str(exc)}
    value["published_monotonic_ns"] = time.monotonic_ns()
    write_status_file(output, value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bar1", type=Path, default=Path("/sys/bus/pci/devices/0000:01:00.0/resource1"))
    parser.add_argument("--port", type=int, default=38262)
    parser.add_argument("--vehicle-status", type=Path, default=Path("/dev/shm/adas_vehicle_status.json"))
    args = parser.parse_args()
    bridge = Bridge(args.bar1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    sock.settimeout(.025)
    next_telemetry = 0
    link_failed = False
    print(json.dumps({"stage": "hardware_bridge_ready", "bar1": str(args.bar1),
                      "port": args.port, "status": bridge.status()}), flush=True)
    while True:
        if time.monotonic() >= next_telemetry:
            if not link_failed:
                telemetry = publish_vehicle(bridge, args.vehicle_status)
                link_failed = "identity/version/caps mismatch" in telemetry.get("error", "")
                if link_failed:
                    print(json.dumps({"stage": "pcie_link_lost", "telemetry": telemetry}), flush=True)
            next_telemetry = time.monotonic() + .025
        try:
            packet, peer = sock.recvfrom(4096)
        except socket.timeout:
            continue
        request = None
        try:
            request = json.loads(packet.decode("utf-8"))
            if link_failed:
                raise OSError("PCIe identity lost; reconnect hardware and restart bridge")
            reply = dispatch(bridge, request)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            reply = {"ok": False, "reason": str(exc), "control_bridge": "ADAS_REAL_PCIE_V1"}
            if isinstance(request, dict) and "request_id" in request:
                reply["request_id"] = request["request_id"]
        if isinstance(request, dict) and request.get("type") == "action" and request.get("action") == "brake":
            vehicle = {"online": False} if link_failed else publish_vehicle(bridge, args.vehicle_status)
            reply["vehicle_feedback"] = vehicle
            print(json.dumps({"stage": "hardware_action", "timestamp_ns": time.monotonic_ns(),
                              "request": request, "reply": reply}), flush=True)
        try:
            sock.sendto(json.dumps(reply, separators=(",", ":")).encode(), peer)
        except OSError:
            pass

if __name__ == "__main__":
    main()
