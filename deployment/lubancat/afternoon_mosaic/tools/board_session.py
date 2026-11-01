"""Start an isolated board test and restore the recorded production session.

Run only on the board. No service files or autostart files are edited.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
STATE = EVIDENCE / "session.json"


def record(pid):
    base = Path(f"/proc/{pid}")
    env = dict(x.split("=", 1) for x in base.joinpath("environ").read_bytes().decode().split("\0")
               if x and x.startswith(("ADAS_", "QT_", "DISPLAY=", "XAUTHORITY=")))
    return {"pid": pid, "exe": os.readlink(base / "exe"),
            "cwd": os.readlink(base / "cwd"), "env": env,
            "args": base.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")}


def stop(pid):
    if not Path(f"/proc/{pid}").exists():
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(40):
        if not Path(f"/proc/{pid}").exists():
            return
        time.sleep(.1)
    os.kill(pid, signal.SIGKILL)


def unlimited_core():
    resource.setrlimit(resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


def launch(args, cwd, env, log):
    with (EVIDENCE / log).open("ab", buffering=0) as stream:
        return subprocess.Popen(args, cwd=cwd, env=dict(os.environ, **env),
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                                start_new_session=True, preexec_fn=unlimited_core).pid


def start(build, label, ai_hz, main_hz):
    assert (ROOT / build / "qt_gpu_probe").is_file(), "Build the HMI before stopping anything"
    if not STATE.exists():
        hmi = [record(int(p)) for p in subprocess.check_output(["pgrep", "-x", "qt_gpu_probe"]).split()]
        assert len(hmi) == 1, "Require exactly one baseline HMI"
        ai = []
        for p in Path("/proc").iterdir():
            if p.name.isdigit():
                try:
                    cmd = (p / "cmdline").read_bytes().split(b"\0")
                    if any(x.endswith(b"b05p_ai_bridge_launcher.py") for x in cmd):
                        ai.append(record(int(p.name)))
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    pass
        assert len(ai) == 1, "Require exactly one baseline AI"
        parent = int(Path(f"/proc/{hmi[0]['pid']}/stat").read_text().split(") ")[1].split()[1])
        parent_cmd = Path(f"/proc/{parent}/cmdline").read_bytes()
        assert b"start-pcie-hmi-gui.sh" in parent_cmd
        state = {"baseline_hmi": hmi[0], "baseline_ai": ai[0], "autostart_parent": parent}
        STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.kill(parent, signal.SIGSTOP)
        stop(hmi[0]["pid"])
        stop(ai[0]["pid"])
        state["mock_pid"] = launch(["python3", str(ROOT / "tools/mock_actions.py"), "38362",
                                     str(EVIDENCE / "mock_actions.jsonl")], ROOT, {}, "mock.log")
    else:
        state = json.loads(STATE.read_text())
        for name in ("test_hmi_pid", "test_ai_pid"):
            if state.get(name):
                stop(state[name])
    ai_args = state["baseline_ai"]["args"][:]
    ai_args[1] = str(ROOT / "tools/isolated_ai.py")
    for flag, value in [("--main-hz", main_hz), ("--road-quadrant", "0"),
                        ("--timing-log", str(EVIDENCE / f"{label}_ai.jsonl")),
                        ("--summary", str(EVIDENCE / f"{label}_ai_summary.json"))]:
        ai_args[ai_args.index(flag) + 1] = str(value)
    env = dict(state["baseline_ai"]["env"], ADAS_EDGE_ROOT=str(ROOT / "ai"))
    state["test_ai_pid"] = launch(ai_args, ROOT / "ai", env, f"{label}_ai.log")
    hmi_args = state["baseline_hmi"]["args"][:]
    hmi_args[0] = str(ROOT / build / "qt_gpu_probe")
    hmi_args[hmi_args.index("--ai-hz") + 1] = str(ai_hz)
    hmi_args += ["--metrics-jsonl", str(EVIDENCE / f"{label}_hmi.jsonl")]
    state["test_hmi_pid"] = launch(hmi_args, ROOT / build, state["baseline_hmi"]["env"], f"{label}_hmi.log")
    state["label"] = label
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state, indent=2), flush=True)


def restore():
    state = json.loads(STATE.read_text())
    for name in ("test_hmi_pid", "test_ai_pid", "mock_pid"):
        if state.get(name):
            stop(state[name])
    base = state["baseline_ai"]
    pid = launch(base["args"], base["cwd"], base["env"], "restored_baseline_ai.log")
    os.kill(state["autostart_parent"], signal.SIGCONT)
    print("restored baseline AI", pid, "and resumed original autostart loop")
    STATE.rename(EVIDENCE / f"session_restored_{time.monotonic_ns()}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "restore"])
    parser.add_argument("--build", default="build-baseline")
    parser.add_argument("--label", default="stage1")
    parser.add_argument("--ai-hz", default="10")
    parser.add_argument("--main-hz", default="10")
    args = parser.parse_args()
    if args.action == "restore":
        restore()
    else:
        start(args.build, args.label, args.ai_hz, args.main_hz)
