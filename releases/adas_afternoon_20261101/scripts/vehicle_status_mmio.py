#!/usr/bin/env python3
import mmap
import os
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/cat/qt_gpu_probe_ai_20261019/pr3_rollback_20261030')
from vehicle_telemetry_v1 import decode, write_status_file, REG, SNAPSHOT

path = '/sys/bus/pci/devices/0000:01:00.0/resource1'
out = Path('/dev/shm/adas_vehicle_status.json')
fd = os.open(path, os.O_RDONLY)
mm = mmap.mmap(fd, 4096, access=mmap.ACCESS_READ)


def rd(off):
    return struct.unpack_from('<I', mm, off)[0]


try:
    while True:
        generation = rd(REG['generation'])
        words = {name: rd(REG[name]) for name in SNAPSHOT}
        words['action_hold_ms'] = rd(REG['action_hold_ms'])
        words['hsst_stale_ms'] = rd(REG['hsst_stale_ms'])
        if generation != rd(REG['generation']):
            time.sleep(0.025)
            continue
        write_status_file(out, decode(words, generation))
        time.sleep(0.025)
finally:
    mm.close()
    os.close(fd)
