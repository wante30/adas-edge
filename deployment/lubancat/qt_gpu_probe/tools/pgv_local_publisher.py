#!/usr/bin/env python3
"""Publish a local 1280x720 RGB565 sequence as PGVF for offline HUD tests.

The input is a raw little-endian RGB565 file (no FPGA).  Transport defaults to
60 fps while the source asset can be 30 fps; duplicate source frames are held for
two transport ticks and the UI must be labelled `pgv-local`.

This is a local replay fixture, not PCIe capture.
"""
from __future__ import annotations

import argparse
import mmap
import os
import struct
import time
import zlib
from pathlib import Path

WIDTH, HEIGHT = 1280, 720
FRAME_BYTES = WIDTH * HEIGHT * 2
HEADER = struct.Struct("<4s7IQ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="raw RGB565LE frames")
    parser.add_argument("--output", required=True, help="latest.pgv output path")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--generation", type=int, default=1)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    size = raw_path.stat().st_size
    if size % FRAME_BYTES:
        raise SystemExit(f"raw size {size} is not a multiple of {FRAME_BYTES}")
    frame_count = size // FRAME_BYTES
    if not frame_count:
        raise SystemExit("raw input has no frames")

    fd = os.open(str(raw_path), os.O_RDONLY)
    mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
    print(f"preparing CRC for {frame_count} frames ...", flush=True)
    crc = []
    for index in range(frame_count):
        crc.append(zlib.crc32(mm[index * FRAME_BYTES:(index + 1) * FRAME_BYTES]) & 0xFFFFFFFF)
    print(f"READY frames={frame_count} fps={args.fps} source_fps={args.source_fps}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name("." + output.name + f".{os.getpid()}.tmp")
    period = 1.0 / max(1.0, args.fps)
    next_tick = time.monotonic()
    transport_index = 0
    last_source_index = -1
    cycle = 0
    while True:
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
            continue
        source_index = int(transport_index * args.source_fps / args.fps) % frame_count
        if last_source_index >= 0 and source_index < last_source_index:
            cycle += 1
        last_source_index = source_index
        offset = source_index * FRAME_BYTES
        stamp = time.monotonic_ns()
        with temp.open("wb") as stream:
            stream.write(HEADER.pack(
                b"PGVF", 1, WIDTH, HEIGHT, transport_index & 0xFFFFFFFF,
                args.generation + cycle, crc[source_index], FRAME_BYTES, stamp))
            stream.write(mm[offset:offset + FRAME_BYTES])
        os.replace(temp, output)
        transport_index += 1
        next_tick += period
        if next_tick < time.monotonic():
            next_tick = time.monotonic() + period
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
