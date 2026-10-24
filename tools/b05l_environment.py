"""Record the exact B-05L PC environment, including the active Windows power plan."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def command(*args: str) -> str | None:
    try:
        output = subprocess.check_output(args, stderr=subprocess.STDOUT, text=True, timeout=20,
                                         encoding="utf-8", errors="replace")
        return output.strip()
    except Exception as exc:  # noqa: BLE001 - environment probing is best-effort
        return f"ERROR: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    import torchvision
    import ultralytics

    info: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "onnxruntime_providers": ort.get_available_providers(),
        "nvidia_smi": command("nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.sm,power.draw",
                              "--format=csv,noheader"),
        "power_plan": command("powercfg", "/getactivescheme"),
        "power_plans": command("powercfg", "/list"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(info, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
