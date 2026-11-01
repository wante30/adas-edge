"""Sample CPU / NPU / GPU / temperature while a benchmark child runs."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path


def read_cpu():
    parts = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values = [int(v) for v in parts]
    return sum(values), values[3] + values[4]


def read_load(path, pattern):
    try:
        text = Path(path).read_text().strip()
    except OSError:
        return None
    match = re.search(pattern, text)
    return match.group(0) if match else text


def main():
    model, prefix = sys.argv[1], sys.argv[2]
    log = open(prefix + ".log", "w")
    proc = subprocess.Popen(
        ["./rknn_bench_fast", model, "dummy.json", "0"],
        stdout=log, stderr=subprocess.STDOUT,
    )
    cpu, npu, gpu, temps = [], [], [], []
    while proc.poll() is None:
        try:
            total, idle = read_cpu()
            cpu.append((total, idle))
        except OSError:
            pass
        value = read_load("/sys/class/devfreq/fde40000.npu/load", r"\d+")
        if value is not None:
            npu.append(int(value))
        value = read_load("/sys/class/devfreq/fde60000.gpu/load", r"\d+")
        if value is not None:
            gpu.append(int(value))
        for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
            try:
                temps.append(int((zone / "temp").read_text()) / 1000.0)
            except (OSError, ValueError):
                pass
        time.sleep(0.1)
    log.close()
    result = {"returncode": proc.returncode}
    if len(cpu) >= 2:
        total = cpu[-1][0] - cpu[0][0]
        idle = cpu[-1][1] - cpu[0][1]
        result["cpu_percent"] = round(100.0 * (1.0 - idle / max(1, total)), 2)
    if npu:
        result.update(npu_mean=round(sum(npu) / len(npu), 2), npu_max=max(npu),
                      npu_samples=len(npu))
    if gpu:
        result.update(gpu_mean=round(sum(gpu) / len(gpu), 2), gpu_max=max(gpu),
                      gpu_samples=len(gpu))
    if temps:
        result.update(temp_min_c=round(min(temps), 2), temp_max_c=round(max(temps), 2),
                      temp_mean_c=round(sum(temps) / len(temps), 2))
    result["note"] = "Qt/HMI was running; values include whole-board load, not model-only."
    Path(prefix + "_resource.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
