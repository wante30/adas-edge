"""Run one isolated HMI/AI session, preserving services and the original window."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
STATE = EVIDENCE / "integration_session.json"
BOOT = Path("/proc/sys/kernel/random/boot_id")
SERVICE = "adas-hazard-scene-live.service"


def record(pid):
    base = Path(f"/proc/{pid}")
    env = dict(x.split("=", 1) for x in base.joinpath("environ").read_bytes().decode().split("\0")
               if x and x.startswith(("ADAS_", "QT_", "DISPLAY=", "XAUTHORITY=")))
    return {"pid": pid, "exe": os.readlink(base / "exe"), "cwd": os.readlink(base / "cwd"),
            "env": env, "args": base.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")}


def stop(pid):
    if not pid or not Path(f"/proc/{pid}").exists():
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not Path(f"/proc/{pid}").exists():
            return
        if Path(f"/proc/{pid}/stat").read_text().split(") ")[1][0] == "Z":
            return
        time.sleep(.1)
    os.kill(pid, signal.SIGKILL)


def launch(args, cwd, env, log):
    with (EVIDENCE / log).open("ab", buffering=0) as stream:
        return subprocess.Popen(args, cwd=cwd, env=dict(os.environ, **env),
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                                start_new_session=True).pid


def save(state):
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def start(args):
    assert (ROOT / args.build / "qt_gpu_probe").is_file()
    if STATE.exists():
        state = json.loads(STATE.read_text())
        assert state["boot_id"] == BOOT.read_text().strip(), "Old boot: preserve and move the old session record first"
        for name in ("test_hmi_pid", "test_ai_pid"):
            stop(state.get(name))
    else:
        hmis = [int(p) for p in subprocess.check_output(["pgrep", "-x", "qt_gpu_probe"]).split()
                if Path(f"/proc/{int(p)}/stat").read_text().split(") ")[1][0] != "Z"]
        assert len(hmis) == 1, f"Require one HMI, got {hmis}"
        hmi = record(hmis[0])
        parent = int(Path(f"/proc/{hmis[0]}/stat").read_text().split(") ")[1].split()[1])
        assert b"start-pcie-hmi-gui.sh" in Path(f"/proc/{parent}/cmdline").read_bytes()
        active = subprocess.call(["systemctl", "is-active", "--quiet", SERVICE]) == 0
        state = {"boot_id": BOOT.read_text().strip(), "baseline_hmi": hmi,
                 "autostart_parent": parent, "ai_service_was_active": active}
        save(state)
        os.kill(parent, signal.SIGSTOP)
        if active:
            subprocess.run(["sudo", "-n", "systemctl", "stop", SERVICE], check=True)
        stop(hmis[0])
        # A separate UDP endpoint receives every test brake decision.
        state["mock_pid"] = launch(["python3", str(ROOT / "tools/mock_actions.py"), "38362",
                                    str(EVIDENCE / "mock_actions.jsonl")], ROOT, {}, "mock.log")
    ai_args = ["/usr/bin/python3", "-u", str(ROOT / "tools/isolated_ai.py"),
               "--live-frame-file", str(ROOT / "evidence/test_frame.mfr1"),
               "--source", "PGV_LIVE", "--layout", "quad", "--display-view", "front",
               "--road-quadrant", "0", "--navigation-quadrant", "1",
               "--main-model", "/home/cat/models/b05n_p2_mtl_b2a_seg256/b05n_p2_mtl_b2a_seg256_int8.rknn",
               "--main-hz", "10", "--hazard-hz", "5", "--ocr-hz", "2", "--navigation-ocr",
               "--idle-timeout", "0", "--timing-log", str(EVIDENCE / f"{args.label}_ai.jsonl"),
               "--summary", str(EVIDENCE / f"{args.label}_ai_summary.json")]
    env = {"ADAS_EDGE_ROOT": str(ROOT / "ai"), "ADAS_DISABLE_ROAD_SCENE": "1",
           "ADAS_LEGACY_HAZARD": "1", "ADAS_AUTO_BRAKE": "1", "ADAS_AI_DEBUG": "1",
           "ADAS_TEST_ACTION_PORT": str(args.action_port)}
    state["test_ai_pid"] = launch(ai_args, ROOT / "ai", env, f"{args.label}_ai.log")
    hmi_args = state["baseline_hmi"]["args"][:]
    hmi_args[0] = str(ROOT / args.build / "qt_gpu_probe")
    hmi_args[hmi_args.index("--ai-hz") + 1] = str(args.ai_hz)
    hmi_args[hmi_args.index("--ai-frame-file") + 1] = str(ROOT / "evidence/test_frame.mfr1")
    hmi_args += ["--metrics-jsonl", str(EVIDENCE / f"{args.label}_hmi.jsonl")]
    state["test_hmi_pid"] = launch(hmi_args, ROOT / args.build, state["baseline_hmi"]["env"], f"{args.label}_hmi.log")
    state["label"] = args.label
    save(state)
    print(json.dumps(state, indent=2), flush=True)


def restore():
    state = json.loads(STATE.read_text())
    assert state["boot_id"] == BOOT.read_text().strip(), "Refuse to signal PIDs from a previous boot"
    for name in ("test_hmi_pid", "test_ai_pid", "mock_pid"):
        stop(state.get(name))
    if state["ai_service_was_active"]:
        subprocess.run(["sudo", "-n", "systemctl", "start", SERVICE], check=True)
    os.kill(state["autostart_parent"], signal.SIGCONT)
    STATE.rename(EVIDENCE / f"integration_session_restored_{time.monotonic_ns()}.json")
    print("Restored original service and autostart state", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "restore"])
    parser.add_argument("--build", default="build-baseline")
    parser.add_argument("--label", default="stage1_after_reboot")
    parser.add_argument("--ai-hz", default="10")
    parser.add_argument("--action-port", default="38362")
    args = parser.parse_args()
    EVIDENCE.mkdir(exist_ok=True)
    start(args) if args.action == "start" else restore()
