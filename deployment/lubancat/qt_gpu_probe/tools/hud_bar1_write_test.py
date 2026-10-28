#!/usr/bin/env python3
"""One-shot V2 + legacy HUD BAR1 write test. Must run as root."""
from __future__ import annotations

import ctypes
import mmap
import os
import sys
import time
from pathlib import Path

PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "/sys/bus/pci/devices/0000:01:00.0/resource1")
MAGIC = 0x31445548

fd = os.open(str(PATH), os.O_RDWR | getattr(os, "O_SYNC", 0))
try:
    mm = mmap.mmap(fd, 4096, access=mmap.ACCESS_WRITE)

    def rd(off):
        return ctypes.c_uint32.from_buffer(mm, off).value

    def wr(off, value):
        ctypes.c_uint32.from_buffer(mm, off).value = value

    print(f"magic=0x{rd(0x600):08x} version=0x{rd(0x604):08x} caps=0x{rd(0x630):08x}")
    if rd(0x600) != MAGIC:
        raise SystemExit("HUD magic mismatch")

    ext = 0x00000A41  # perception, left solid, right dashed, LKA normal
    wr(0x638, ext)
    wr(0x63C, 0)
    wr(0x640, 0)
    seq = 0x1234
    wr(0x644, seq)
    deadline = time.monotonic() + 0.2
    status = 0
    while time.monotonic() < deadline:
        status = rd(0x648)
        if not status & 1 and status & 2:
            break
        time.sleep(0.001)
    print(f"ext_request=0x{rd(0x638):08x} ext_status=0x{status:08x} applied=0x{rd(0x64C):08x}")

    legacy = (0x2345 << 16) | (1 << 12) | 0x101  # hud + adas visible, override
    wr(0x608, legacy)
    time.sleep(0.02)
    print(f"legacy=0x{legacy:08x} accepted=0x{rd(0x610):08x} applied=0x{rd(0x614):08x} "
          f"rk=0x{rd(0x618):08x} effective=0x{rd(0x61C):08x} "
          f"reject=0x{rd(0x620):08x} err=0x{rd(0x624):08x}")
    mm.close()
finally:
    os.close(fd)
