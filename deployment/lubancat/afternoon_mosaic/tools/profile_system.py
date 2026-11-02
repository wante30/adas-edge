"""Monotonic real PCIe sampler; /proc CPU is normalized to all cores."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"


def proc_sample():
    total = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:]))
    tasks = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            for task in proc.joinpath("task").iterdir():
                raw = task.joinpath("stat").read_text()
                fields = raw.rsplit(") ", 1)[1].split()
                sched = list(map(int, task.joinpath("schedstat").read_text().split()))
                key = proc.name + ":" + task.name
                tasks[key] = dict(pid=int(proc.name), tid=int(task.name),
                                  name=raw.split("(", 1)[1].rsplit(")", 1)[0], cmd=cmd[:500],
                                  start_ticks=int(fields[19]), ticks=int(fields[11])+int(fields[12]),
                                  run_ns=sched[0], wait_ns=sched[1])
        except (OSError, ValueError, IndexError):
            continue
    return dict(timestamp_ns=time.monotonic_ns(), cpu=total, tasks=tasks)


def read_new(path, offset):
    if not path.exists():
        return [], offset
    with path.open("rb") as f:
        f.seek(offset)
        rows = []
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.endswith(b"\n"):
                return rows, pos
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass


def sample(label, seconds):
    EVIDENCE.mkdir(exist_ok=True)
    state = json.loads((EVIDENCE/"integration_session.json").read_text())
    pid = state.get("test_ai_pid")
    if pid:
        env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        assert b"ADAS_CONTROL_AUTHORIZED=0" in env, "Performance requires actuator sends OFF"
    paths = {key: EVIDENCE/(state["label"]+"_"+key+".jsonl") for key in ("hmi", "ai")}
    offsets = {key: (p.stat().st_size if p.exists() else 0) for key,p in paths.items()}
    previous = proc_sample()
    deadline = time.monotonic()+seconds
    with (EVIDENCE/(label+"_system.jsonl")).open("w", buffering=1) as out:
        out.write(json.dumps(dict(kind="metadata", session=state, seconds=seconds,
                                  boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip()))+"\n")
        while time.monotonic()<deadline:
            time.sleep(1)
            current=proc_sample()
            dt=(current["timestamp_ns"]-previous["timestamp_ns"])/1e9
            delta=[b-a for a,b in zip(previous["cpu"], current["cpu"])]
            total=sum(delta[:8]); idle=delta[3]+delta[4]
            threads=[]
            for key,t in current["tasks"].items():
                old=previous["tasks"].get(key)
                if not old or old["start_ticks"] != t["start_ticks"]:
                    continue
                cpu=(t["ticks"]-old["ticks"])/os.sysconf("SC_CLK_TCK")/dt*100
                if cpu or (t["wait_ns"]-old["wait_ns"])>1000000:
                    threads.append(dict(pid=t["pid"],tid=t["tid"],name=t["name"],cmd=t["cmd"],
                        cpu_one_core_percent=cpu, run_ms=(t["run_ns"]-old["run_ns"])/1e6,
                        runqueue_wait_ms=(t["wait_ns"]-old["wait_ns"])/1e6))
            row=dict(kind="sample", timestamp_ns=current["timestamp_ns"],dt_s=dt,
                     system_cpu_percent=100*(1-idle/total) if total else None,
                     threads=threads)
            for key,p in paths.items():
                row[key],offsets[key]=read_new(p,offsets[key])
            for key,p in (("capture",Path("/dev/shm/pango-video/status.json")),
                          ("lvt1",Path("/dev/shm/adas_vehicle_status.json"))):
                try: row[key]=json.loads(p.read_text())
                except (OSError,ValueError): row[key]={}
            out.write(json.dumps(row,separators=(",",":"))+"\n")
            previous=current
    print("PROFILE_DONE", label, flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--seconds", type=int, default=45)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--build", default="build-stage4")
    args=parser.parse_args()
    if args.matrix:
        for name,flags in (("C",[]),("B",["--no-navigation-ocr"]),
                           ("A",["--no-ai"]),("D",["--diagnostic-static-tesla"])):
            label=args.label+"_"+name
            subprocess.run(["python3",str(ROOT/"tools/test_session.py"),"start",
                "--build",args.build,"--label",label,"--ai-hz","0","--control-mode","HARDWARE",
                "--external-qml"]+flags,check=True)
            time.sleep(25)
            sample(label,args.seconds)
    else:
        sample(args.label,args.seconds)
