"""Convert the RK3566 X11 root-window dump without ImageMagick."""
import argparse
from pathlib import Path
import struct
from PIL import Image


def convert(source, destination):
    data = Path(source).read_bytes()
    h = struct.unpack(">25I", data[:100])
    assert h[1] == 7 and h[2] == 2 and h[11] == 32 and h[7] == 0
    assert h[14:17] == (0xFF0000, 0xFF00, 0xFF), "Unsupported X11 pixel masks"
    offset = h[0] + h[19] * 12
    image = Image.frombytes("RGB", (h[4], h[5]), data[offset:], "raw", "BGRX", h[12], 1)
    image.save(destination)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("source"); p.add_argument("destination")
    a = p.parse_args(); convert(a.source, a.destination)
