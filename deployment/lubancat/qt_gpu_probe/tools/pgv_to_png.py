#!/usr/bin/env python3
"""Convert one PGVF latest.pgv snapshot to PNG for layout inspection."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import cv2
import numpy as np

PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/pgv_qt_hud/latest.pgv")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/pgv_frame.png")
HEADER = struct.Struct("<4s7IQ")
WIDTH, HEIGHT = 1280, 720
FRAME_BYTES = WIDTH * HEIGHT * 2

blob = PATH.read_bytes()
if len(blob) != HEADER.size + FRAME_BYTES:
    raise SystemExit(f"bad PGVF size: {len(blob)}")
magic, version, width, height, frame_id, generation, crc, length, timestamp_ns = HEADER.unpack_from(blob)
if (magic, version, width, height, length) != (b"PGVF", 1, WIDTH, HEIGHT, FRAME_BYTES):
    raise SystemExit("bad PGVF header")
payload = np.frombuffer(blob, dtype=np.uint8, count=FRAME_BYTES, offset=HEADER.size).reshape(HEIGHT, WIDTH, 2)
image = cv2.cvtColor(payload, cv2.COLOR_BGR5652BGR)
cv2.imwrite(str(OUT), image)
print(f"frame_id={frame_id} generation={generation} timestamp_ns={timestamp_ns} -> {OUT}")
