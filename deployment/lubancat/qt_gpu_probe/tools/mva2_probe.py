"""Collect MVA2 packets from a loopback AI bridge for a bounded interval."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import socket
import statistics
import struct
import time


HEADER = struct.Struct("<4sQQIIB3xfI")
BOX = struct.Struct("<7fIf")


def collect(port: int, duration_s: float) -> dict[str, object]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(min(0.5, duration_s))
    started = time.monotonic()
    packets = []
    malformed = 0
    while time.monotonic() - started < duration_s:
        try:
            payload, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        if len(payload) < HEADER.size or payload[:4] != b"MVA2":
            malformed += 1
            continue
        magic, frame_id, timestamp_ns, box_count, flags, nav, distance, status_size = HEADER.unpack_from(payload)
        expected = HEADER.size + box_count * BOX.size + status_size
        if expected != len(payload) or box_count > 32:
            malformed += 1
            continue
        offset = HEADER.size + box_count * BOX.size
        try:
            status = json.loads(payload[offset:].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            malformed += 1
            continue
        classes = Counter()
        for index in range(box_count):
            box = BOX.unpack_from(payload, HEADER.size + index * BOX.size)
            classes[str(int(box[7]))] += 1
        packets.append({
            "frame_id": int(frame_id),
            "timestamp_ns": int(timestamp_ns),
            "boxes": int(box_count),
            "class_ids": dict(classes),
            "hazards": status.get("hazards", []),
            "hazard_details": status.get("hazard_details", []),
        })
    sock.close()
    elapsed = max(time.monotonic() - started, 1e-9)
    intervals_ms = [
        (right["timestamp_ns"] - left["timestamp_ns"]) / 1e6
        for left, right in zip(packets, packets[1:])
        if right["timestamp_ns"] >= left["timestamp_ns"]
    ]
    return {
        "schema": "mva2-probe-v1",
        "port": port,
        "duration_s": elapsed,
        "valid_packets": len(packets),
        "packet_fps": len(packets) / elapsed,
        "malformed_packets": malformed,
        "mean_boxes": statistics.fmean(item["boxes"] for item in packets) if packets else None,
        "frame_interval_ms_p50": statistics.median(intervals_ms) if intervals_ms else None,
        "packets": packets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=39261)
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()
    if not 0 < args.duration <= 300:
        raise SystemExit("duration must be in (0, 300]")
    print(json.dumps(collect(args.port, args.duration), indent=2), flush=True)


if __name__ == "__main__":
    main()
