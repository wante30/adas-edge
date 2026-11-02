"""Run one isolated HMI/AI session, preserving services and the original window."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
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
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(30):
        if not Path(f"/proc/{pid}").exists():
            return
        try:
            if Path(f"/proc/{pid}/stat").read_text().split(") ")[1][0] == "Z":
                return
        except FileNotFoundError:
            return
        time.sleep(.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def launch(args, cwd, env, log):
    with (EVIDENCE / log).open("ab", buffering=0) as stream:
        return subprocess.Popen(args, cwd=cwd, env=dict(os.environ, **env),
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                                start_new_session=True).pid


def save(state):
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def resume_telemetry(state):
    paused = state.pop("paused_vehicle_pid", None)
    if paused and Path(f"/proc/{paused}/cmdline").exists():
        if b"vehicle_status_mmio.py" in Path(f"/proc/{paused}/cmdline").read_bytes():
            subprocess.run(["sudo", "-n", "kill", "-CONT", str(paused)], check=True)


def hardware_bridge(state, enable):
    unit = "adas-risk-hardware.service"
    original = "adas-hud-bridge.service"
    if not enable:
        if "hud_service_was_active" in state:
            subprocess.run(["sudo", "-n", "systemctl", "stop", unit], check=True)
            if state.pop("hud_service_was_active"):
                subprocess.run(["sudo", "-n", "systemctl", "start", original], check=True)
            save(state)
        return
    if "hud_service_was_active" not in state:
        state["hud_service_was_active"] = subprocess.run(
            ["systemctl", "is-active", "--quiet", original]).returncode == 0
        save(state)
    subprocess.run(["sudo", "-n", "systemctl", "stop", original], check=True)
    if subprocess.run(["systemctl", "is-active", "--quiet", unit]).returncode != 0:
        subprocess.run(["sudo", "-n", "systemd-run", "--unit=" + unit, "--collect",
            "--property=Restart=on-failure", "--property=RestartSec=2", "/usr/bin/python3", "-u",
            str(ROOT / "hardware/hardware_bridge.py")], check=True)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.settimeout(.3)
        for _ in range(20):
            probe.sendto(b'{"type":"status"}', ("127.0.0.1", 38262))
            try:
                reply = json.loads(probe.recv(4096))
                if reply.get("control_bridge") == "ADAS_REAL_PCIE_V1":
                    return
            except (socket.timeout, ValueError):
                pass
        raise RuntimeError("Real PCIe bridge did not become ready; inspect journalctl -u " + unit)


def start(args):
    assert (ROOT / args.build / "qt_gpu_probe").is_file()
    if STATE.exists():
        old = json.loads(STATE.read_text())
        if old["boot_id"] != BOOT.read_text().strip():
            STATE.rename(EVIDENCE / ("integration_session_previous_boot_" + old["boot_id"] + "_" + str(time.monotonic_ns()) + ".json"))
    if STATE.exists():
        state = json.loads(STATE.read_text())
        for name in ("test_hmi_pid", "test_ai_pid"):
            stop(state.get(name))
    else:
        hmis = [int(p) for p in subprocess.run(["pgrep", "-x", "qt_gpu_probe"], stdout=subprocess.PIPE).stdout.split()
                if Path(f"/proc/{int(p)}/stat").read_text().split(") ")[1][0] != "Z"]
        assert len(hmis) <= 1, f"Require at most one HMI, got {hmis}"
        parent = 0
        if hmis:
            hmi = record(hmis[0])
            parent = int(Path(f"/proc/{hmis[0]}/stat").read_text().split(") ")[1].split()[1])
            assert b"start-pcie-hmi-gui.sh" in Path(f"/proc/{parent}/cmdline").read_bytes()
        else:
            # An absent PCIe endpoint may keep the formal GUI autostart waiting.
            # Reuse saved arguments/environment, never saved PIDs from another boot.
            history = sorted(EVIDENCE.glob("integration_session_restored_*.json"),
                             key=lambda p: p.stat().st_mtime_ns)
            launch_profile = ROOT / "config/baseline_launch.json"
            assert args.raw_replay and (history or launch_profile.is_file()), "No baseline HMI or launch profile"
            hmi = (json.loads(history[-1].read_text())["baseline_hmi"] if history
                   else json.loads(launch_profile.read_text(encoding="utf-8-sig")))
            for candidate in Path("/proc").glob("[0-9]*/cmdline"):
                try:
                    argv = candidate.read_bytes().split(b"\0")
                    if len(argv) > 1 and argv[1] == b"/home/cat/.local/bin/start-pcie-hmi-gui.sh":
                        parent = int(candidate.parent.name)
                        break
                except (OSError, ValueError):
                    pass
        service_state = subprocess.run(["systemctl", "is-active", SERVICE],
                                       stdout=subprocess.PIPE, text=True).stdout.strip()
        active = service_state in ("active", "activating")
        state = {"boot_id": BOOT.read_text().strip(), "baseline_hmi": hmi,
                 "autostart_parent": parent, "ai_service_was_active": active}
        save(state)
        if parent:
            os.kill(parent, signal.SIGSTOP)
        if active:
            subprocess.run(["sudo", "-n", "systemctl", "stop", SERVICE], check=True)
        stop(hmis[0] if hmis else None)
    if not args.raw_replay:
        resume_telemetry(state)
    forward_quadrant = getattr(args,"forward_quadrant",None) if getattr(args,"forward_quadrant",None) is not None else (0 if args.raw_replay else 3)
    shared_frame = Path("/dev/shm") / (ROOT.name + ".mfr1")
    ai_args = ["/usr/bin/python3", "-u", str(ROOT / "tools/isolated_ai.py"),
               "--live-frame-file", str(shared_frame),
               "--source", "LOCAL_REPLAY" if args.raw_replay else "PGV_LIVE", "--layout", "quad", "--display-view", "front",
               "--road-quadrant", str(forward_quadrant), "--forward-quadrant", str(forward_quadrant), "--navigation-quadrant", "1",
               "--main-model", "/home/cat/models/b05n_p2_mtl_b2a_seg256/b05n_p2_mtl_b2a_seg256_int8.rknn",
               "--main-hz", "10", "--hazard-hz", "5", "--ocr-hz", "2", "--navigation-ocr",
               "--idle-timeout", "0", "--timing-log", str(EVIDENCE / f"{args.label}_ai.jsonl"),
               "--summary", str(EVIDENCE / f"{args.label}_ai_summary.json")]
    env = {"ADAS_EDGE_ROOT": str(ROOT / "ai"), "ADAS_DISABLE_ROAD_SCENE": "1",
           "ADAS_LEGACY_HAZARD": "1", "ADAS_AUTO_BRAKE": "1", "ADAS_AI_DEBUG": "1",
           "ADAS_TEST_ACTION_PORT": str(args.action_port),
           "ADAS_CONTROL_MODE": getattr(args, "control_mode", None) or ("DRY_RUN" if args.raw_replay else "HARDWARE"),
           "ADAS_EVENT_TIMELINE": str(EVIDENCE / (args.label + "_events.jsonl"))}
    if getattr(args,"synthetic_ui",False):
        if env["ADAS_CONTROL_MODE"]!="DRY_RUN":raise ValueError("Synthetic fixture requires DRY_RUN")
        ai_args=["python3",str(ROOT/"tools/synthetic_ui.py")]
    hardware_bridge(state, env["ADAS_CONTROL_MODE"] == "HARDWARE")
    state["control_mode"] = env["ADAS_CONTROL_MODE"]
    state["forward_quadrant"] = forward_quadrant
    state["input_kind"] = "LOCAL_REPLAY" if args.raw_replay else "REAL_PCIE"
    state["test_ai_pid"] = launch(ai_args, ROOT / "ai", env, f"{args.label}_ai.log")
    hmi_args = state["baseline_hmi"]["args"][:]
    hmi_args[0] = str(ROOT / args.build / "qt_gpu_probe")
    if "--road-quadrant" in hmi_args:
        hmi_args[hmi_args.index("--road-quadrant") + 1] = str(forward_quadrant)
    else:
        hmi_args += ["--road-quadrant", str(forward_quadrant)]
    hmi_args += ["--forward-quadrant", str(forward_quadrant)]
    hmi_args[hmi_args.index("--ai-hz") + 1] = str(args.ai_hz)
    hmi_args[hmi_args.index("--ai-frame-file") + 1] = str(shared_frame)
    if args.raw_replay:
        for flag in ("--pgv-file", "--source-label"):
            if flag in hmi_args:
                i = hmi_args.index(flag)
                del hmi_args[i:i + 2]
        hmi_args += ["--raw-replay", args.raw_replay]
    hmi_args += ["--metrics-jsonl", str(EVIDENCE / f"{args.label}_hmi.jsonl")]
    hmi_env = dict(state["baseline_hmi"]["env"])
    hmi_env["ADAS_CONTROL_MODE"] = env["ADAS_CONTROL_MODE"]
    hmi_env["QT_RISK_TIMELINE_LOG"] = str(EVIDENCE / (args.label + "_risk_ui.jsonl"))
    hmi_env["QT_AI_PACING_POLICY"] = os.environ.get("QT_AI_PACING_POLICY", "PRESSURE")
    hmi_env["QT_AI_SCHEDULE_LOG"] = str(EVIDENCE / (args.label + "_schedule.jsonl"))
    if os.environ.get("QT_GUI_PROFILE"): hmi_env["QT_GUI_PROFILE"] = "1"
    hmi_env["QT_SCENE_TIMELINE_LOG"] = str(EVIDENCE / (args.label + "_scene.jsonl"))
    hmi_env["QT_AI_TIMING_LOG"] = str(EVIDENCE / (args.label + "_qt_ai.jsonl"))
    if args.external_qml:
        hmi_env["QT_QML_FILE"] = str(ROOT / "hmi/main.qml")
    state["test_hmi_pid"] = launch(hmi_args, ROOT / args.build, hmi_env, f"{args.label}_hmi.log")
    state["label"] = args.label
    save(state)
    print(json.dumps(state, indent=2), flush=True)


def restore():
    state = json.loads(STATE.read_text())
    assert state["boot_id"] == BOOT.read_text().strip(), "Refuse to signal PIDs from a previous boot"
    for name in ("test_hmi_pid", "test_ai_pid", "mock_pid"):
        stop(state.get(name))
    resume_telemetry(state)
    hardware_bridge(state, False)
    if state["ai_service_was_active"]:
        subprocess.run(["sudo", "-n", "systemctl", "start", SERVICE], check=True)
    if state.get("autostart_parent"):
        os.kill(state["autostart_parent"], signal.SIGCONT)
    STATE.rename(EVIDENCE / f"integration_session_restored_{time.monotonic_ns()}.json")
    print("Restored original service and autostart state", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "restore"])
    parser.add_argument("--build", default="build-baseline")
    parser.add_argument("--label", default="stage1_after_reboot")
    parser.add_argument("--forward-quadrant", type=int, choices=range(4))
    parser.add_argument("--ai-hz", default="10")
    parser.add_argument("--external-qml", action="store_true")
    parser.add_argument("--synthetic-ui", action="store_true")
    parser.add_argument("--raw-replay", default="")
    parser.add_argument("--action-port", default="38262")
    parser.add_argument("--control-mode", choices=["DRY_RUN", "HARDWARE"])
    args = parser.parse_args()
    EVIDENCE.mkdir(exist_ok=True)
    start(args) if args.action == "start" else restore()
