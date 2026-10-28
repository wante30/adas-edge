#!/usr/bin/env python3
"""Preloaded RAM PGVF publisher for local replay benchmarks.

120 distinct RGB565 frames are read once at startup and then published from
memory.  No per-frame disk read, PIL/OpenCV decode, or CRC is involved.
"""
from __future__ import annotations

import argparse
import os
import struct
import time
from pathlib import Path

WIDTH, HEIGHT = 1280, 720
FRAME_BYTES = WIDTH * HEIGHT * 2
HEADER = struct.Struct("<4s7IQ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--stats", default="")
    args = parser.parse_args()
    if args.fps <= 0 or args.count <= 0:
        raise SystemExit("fps/count must be positive")

    raw = Path(args.raw)
    with raw.open("rb") as stream:
        stream.seek(args.start_frame * FRAME_BYTES)
        frames = [stream.read(FRAME_BYTES) for _ in range(args.count)]
    if any(len(frame) != FRAME_BYTES for frame in frames):
        raise SystemExit("raw input ended before the requested frame count")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name("." + output.name + f".{os.getpid()}.tmp")
    period = 1.0 / args.fps
    start = time.monotonic()
    deadline = start + args.seconds if args.seconds > 0 else None
    next_tick = start
    index = 0
    generation = 0
    published = 0
    last_report = start
    print(f"READY ram_frames={len(frames)} fps={args.fps} bytes={len(frames)*FRAME_BYTES}", flush=True)
    try:
        while deadline is None or time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.001, next_tick - now))
                continue
            frame = frames[index % len(frames)]
            if index and index % len(frames) == 0:
                generation += 1
            stamp = time.monotonic_ns()
            with temp.open("wb") as stream:
                stream.write(HEADER.pack(
                    b"PGVF", 1, WIDTH, HEIGHT, published & 0xFFFFFFFF,
                    generation, 0, FRAME_BYTES, stamp))
                stream.write(frame)
            os.replace(temp, output)
            published += 1
            index += 1
            next_tick += period
            after = time.monotonic()
            if next_tick <= after:
                # Do not accumulate unbounded error after a long stall.
                next_tick = after + period
            if after - last_report >= 1.0:
                actual = (published - 0) / max(1e-9, after - start)
                if args.stats:
                    with open(args.stats, "a", encoding="utf-8") as log:
                        log.write(f"{after-start:.3f} published={published} fps={actual:.3f}\n")
                last_report = after
    except KeyboardInterrupt:
        pass
    elapsed = max(1e-9, time.monotonic() - start)
    print(f"DONE published={published} elapsed={elapsed:.3f} fps={published/elapsed:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
