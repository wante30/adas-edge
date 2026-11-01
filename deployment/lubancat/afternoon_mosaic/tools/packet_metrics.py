"""Read captured MVA2 packets without extra Python packages."""
import json
from pathlib import Path
import struct
import sys


def packets(path):
    data = Path(path).read_bytes()
    endian = "<" if data[:4] == b"\xd4\xc3\xb2\xa1" else ">"
    offset = 24
    while offset + 16 <= len(data):
        sec, usec, size, _ = struct.unpack_from(endian + "IIII", data, offset)
        offset += 16
        frame = data[offset:offset + size]
        offset += size
        if len(frame) < 42 or frame[12:14] != b"\x08\x00":
            continue
        ip = 14
        udp = ip + (frame[ip] & 15) * 4
        if frame[ip + 9] != 17:
            continue
        payload = frame[udp + 8:]
        if payload[:4] != b"MVA2" or len(payload) < 40:
            continue
        frame_id, timestamp, count = struct.unpack_from("<QQI", payload, 4)
        status_bytes = struct.unpack_from("<I", payload, 36)[0]
        start = 40 + count * 36
        if len(payload) != start + status_bytes:
            continue
        status = json.loads(payload[start:])
        boxes = [struct.unpack_from("<7fIf", payload, 40 + i * 36) for i in range(count)]
        yield {"time": sec + usec / 1e6, "frame_id": frame_id,
               "timestamp_ns": timestamp, "boxes": boxes, "status": status}


if __name__ == "__main__":
    rows = list(packets(sys.argv[1]))
    unique = {r["frame_id"] for r in rows}
    span = rows[-1]["time"] - rows[0]["time"] if len(rows) > 1 else 0
    print(json.dumps({"packets": len(rows), "unique_frame_ids": len(unique),
                      "completed_result_fps": (len(unique) - 1) / span if span else 0,
                      "first": rows[0] if rows else None,
                      "last": rows[-1] if rows else None}, ensure_ascii=False, indent=2))
