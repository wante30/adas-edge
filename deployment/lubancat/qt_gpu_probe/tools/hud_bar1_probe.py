#!/usr/bin/env python3
"""Read-only probe of the FPGA HUD BAR1 window. Must run as root."""
from __future__ import annotations

import ctypes
import mmap
import os
import sys
from pathlib import Path

PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "/sys/bus/pci/devices/0000:01:00.0/resource1")
OFFSETS = {
    "HUD_MAGIC": 0x600,
    "HUD_VERSION": 0x604,
    "HUD_STATUS": 0x60C,
    "HUD_CAPS": 0x630,
    "HUD_EXT_REQUEST": 0x638,
    "HUD_EXT_STATUS": 0x648,
    "HUD_EXT_APPLIED_SEQ": 0x64C,
    "HUD_REJECT_COUNT": 0x620,
    "HUD_LAST_ERROR": 0x624,
}

fd = os.open(str(PATH), os.O_RDWR | getattr(os, "O_SYNC", 0))
try:
    mapping = mmap.mmap(fd, 4096, access=mmap.ACCESS_WRITE)
    for name, offset in OFFSETS.items():
        value = ctypes.c_uint32.from_buffer(mapping, offset).value
        print(f"{name:22s} +0x{offset:03x} = 0x{value:08x}")
    mapping.close()
finally:
    os.close(fd)
