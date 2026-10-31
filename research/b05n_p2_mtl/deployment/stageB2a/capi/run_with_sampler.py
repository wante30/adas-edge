import json
import re
import subprocess
import sys
import time

model, prefix = sys.argv[1], sys.argv[2]
log = open(prefix + ".log", "w")
proc = subprocess.Popen(["./rknn_bench", model, "dummy.json", "0"], stdout=log, stderr=subprocess.STDOUT)
cpu = []
npu = []
while proc.poll() is None:
    with open("/proc/stat") as f:
        p = f.readline().split()
    v = list(map(int, p[1:]))
    cpu.append((sum(v), v[3] + v[4]))
    try:
        with open("/sys/kernel/debug/rknpu/load") as f:
            m = re.search(r"(\d+)%", f.read())
            if m:
                npu.append(int(m.group(1)))
    except Exception:
        pass
    time.sleep(0.1)
log.close()
res = {"returncode": proc.returncode}
if len(cpu) >= 2:
    res["cpu_percent"] = 100.0 * (1 - (cpu[-1][1] - cpu[0][1]) / max(1, cpu[-1][0] - cpu[0][0]))
if npu:
    res["npu_mean"] = sum(npu) / len(npu)
    res["npu_max"] = max(npu)
    res["npu_samples"] = len(npu)
open(prefix + "_resource.json", "w").write(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
