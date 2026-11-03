"""Record actual Qt renderers on the board with identical saved RKNN inputs.

The isolated players do not connect to control. The live HMI is resumed in
finally; its AI and hardware services are not reconfigured by this tool.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

BOOT = Path("/proc/sys/kernel/random/boot_id")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interrupted(signum, frame):
    raise KeyboardInterrupt("Interrupted by signal " + str(signum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--raw-video", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path(
        "/home/cat/adas_issue9_20261102/src/evidence/integration_session.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = json.loads(args.state.read_text())
    pid = state["test_hmi_pid"]
    proc = Path("/proc") / str(pid)
    boot = BOOT.read_text().strip()
    assert state["boot_id"] == boot
    original_cmdline = proc.joinpath("cmdline").read_bytes()
    assert b"qt_gpu_probe" in original_cmdline
    receipt = {"boot_id": boot, "measurement_sha256": digest(args.measurements),
               "measurement_path": str(args.measurements),
               "raw_video": str(args.raw_video), "runs": [], "complete": False,
               "mode": "RECORDED_RKNN_RENDERER_ONLY", "actuation": False}
    signal.signal(signal.SIGTERM, interrupted)
    variants = ([("before", args.baseline)] if args.baseline else []) + [("after", args.candidate)]
    os.kill(pid, signal.SIGSTOP)
    try:
        for label, binary in variants:
            run = {"label": label, "binary": str(binary), "binary_sha256": digest(binary)}
            receipt["runs"].append(run)
            env = dict(os.environ, DISPLAY=":0.0", XAUTHORITY="/home/cat/.Xauthority",
                       QT_QPA_PLATFORM="xcb", QT_XCB_GL_INTEGRATION="xcb_egl",
                       QT_BEV_TRACE=str(args.output / (label + "_trace.jsonl")),
                       QT_SCENE_TIMELINE_LOG=str(args.output / (label + "_scene.jsonl")))
            player = recorder = None
            try:
                run["launch_monotonic_ns"] = time.monotonic_ns()
                with (args.output / (label + "_player.log")).open("wb") as log:
                    player = subprocess.Popen([str(binary), str(args.measurements),
                        str(args.raw_video), "RECORDED RKNN | " + label], env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                with (args.output / (label + "_recorder.log")).open("wb") as log:
                    recorder = subprocess.Popen(["ffmpeg", "-n", "-nostdin", "-loglevel", "error",
                        "-f", "x11grab", "-framerate", "12", "-video_size", "1280x720",
                        "-i", ":0.0", "-c:v", "libx264", "-preset", "ultrafast",
                        "-crf", "20", "-threads", "1", "-pix_fmt", "yuv420p",
                        str(args.output / (label + ".mp4"))], env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                run["player_exit"] = player.wait(timeout=95)
                if run["player_exit"]:
                    raise RuntimeError("Replay failed: " + label)
            finally:
                if player and player.poll() is None:
                    player.terminate()
                    player.wait(timeout=5)
                if recorder and recorder.poll() is None:
                    recorder.send_signal(signal.SIGINT)
                    recorder.wait(timeout=15)
                run["recorder_exit"] = recorder.returncode if recorder else None
                run["end_monotonic_ns"] = time.monotonic_ns()
            if run["recorder_exit"] not in (0, 255):
                raise RuntimeError("Screen recording failed: " + label)
            print(label + " complete", flush=True)
        receipt["complete"] = True
    finally:
        if (BOOT.read_text().strip() == boot and proc.exists()
                and proc.joinpath("cmdline").read_bytes() == original_cmdline):
            os.kill(pid, signal.SIGCONT)
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
