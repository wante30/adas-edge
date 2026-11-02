"""Avoid qmake restart loops when extracted sources are newer than board wall time."""
import os
import time
from pathlib import Path
root = Path(__file__).resolve().parents[1]
now = time.time()
for folder in (root / "hmi", root / "fast_mosaic/capi", root / "build-stage4"):
    for path in folder.rglob("*"):
        if path.is_file() and path.stat().st_mtime > now:
            generated = folder.name == "build-stage4" or path.suffix in (".o", ".so")
            stamp = 1 if generated else now
            os.utime(path, (stamp, stamp))
