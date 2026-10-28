#!/usr/bin/env python3
"""Publish a valid PGVF snapshot loop for testing the Qt consumer path.

This is a test fixture, not a PCIe capture. The UI must be labelled with
`--source-label pgv-fixture` when this publisher is used.
"""
from __future__ import annotations

import argparse
import os
import struct
import time
import zlib
from pathlib import Path

WIDTH = 1280
HEIGHT = 720
PAYLOAD = WIDTH * HEIGHT * 2
HEADER = struct.Struct("<4s7IQ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True, help="1280x720 RGB565 raw frame")
    parser.add_argument("--output", required=True, help="latest.pgv path")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs forever")
    parser.add_argument("--generation", type=int, default=1)
    args = parser.parse_args()

    payload = Path(args.frame).read_bytes()
    if len(payload) != PAYLOAD:
        raise SystemExit(f"frame must be {PAYLOAD} bytes, got {len(payload)}")
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name("." + output.name + f".{os.getpid()}.tmp")
    period = 1.0 / max(1.0, args.fps)
    begin = time.monotonic()
    deadline = begin + args.duration if args.duration > 0 else None

    frame_id = 0
    next_frame = time.monotonic()
    with temp.open("wb") as stream:
        while deadline is None or time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_frame:
                time.sleep(min(0.002, next_frame - now))
                continue
            stamp = time.monotonic_ns()
            stream.seek(0)
            stream.write(HEADER.pack(b"PGVF", 1, WIDTH, HEIGHT, frame_id, args.generation,
                                     crc, PAYLOAD, stamp))
            stream.write(payload)
            stream.flush()
            os.replace(temp, output)
            # Recreate the next temporary file; the rename consumes this inode.
            stream.close()
            temp = output.with_name("." + output.name + f".{os.getpid()}.tmp")
            stream = temp.open("wb")
            frame_id = (frame_id + 1) & 0xFFFFFFFF
            next_frame += period
            now = time.monotonic()
            if next_frame < now:
                next_frame = now + period
    try:
        temp.unlink()
    except FileNotFoundError:
        pass
    elapsed = max(0.001, time.monotonic() - begin)
    print(f"published {frame_id} frames in {elapsed:.2f}s ({frame_id / elapsed:.2f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
