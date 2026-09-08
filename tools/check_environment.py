"""Read-only dependency report; it does not install packages or touch hardware."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys


def main() -> None:
    packages = ["cv2", "numpy", "psutil", "ultralytics", "rknn", "rknnlite"]
    print(json.dumps({
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: bool(importlib.util.find_spec(name)) for name in packages},
    }, indent=2))


if __name__ == "__main__":
    main()
