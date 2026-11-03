"""Restore the HMI saved by the lifecycle deployment, retaining AI/control."""
import argparse
import importlib.util
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--session-tool", type=Path, default=Path(
        "/home/cat/adas_issue9_20261102/src/tools/test_session.py"))
    args = parser.parse_args()
    root = args.deployment.resolve()
    receipt = json.loads((root / "deployment_receipt.json").read_text())
    previous = json.loads((root / "previous_hmi.json").read_text())
    spec = importlib.util.spec_from_file_location("session", args.session_tool)
    session = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(session)
    state = json.loads(session.STATE.read_text())
    boot = session.BOOT.read_text().strip()
    if state["boot_id"] != boot or receipt["boot_id"] != boot:
        raise RuntimeError("Different boot: use a fresh session; never signal saved PIDs")
    current = session.record(state["test_hmi_pid"])
    if Path(current["exe"]).resolve() != (root / "build/qt_gpu_probe").resolve():
        raise RuntimeError("Current HMI is not this deployment; refusing to replace it")
    if not Path(previous["args"][0]).is_file():
        raise RuntimeError("Saved HMI binary is missing")
    session.EVIDENCE = Path("/dev/shm") / ("adas_lifecycle_rollback_" + str(time.monotonic_ns()))
    session.EVIDENCE.mkdir()
    session.stop(current["pid"])
    state["test_hmi_pid"] = session.launch(previous["args"], previous["cwd"],
                                          previous["env"], "restored_hmi.log")
    state.pop("lifecycle_deployment", None)
    session.save(state)
    print(json.dumps({"restored_hmi_pid": state["test_hmi_pid"],
                      "unchanged_ai_pid": state["test_ai_pid"],
                      "binary": previous["args"][0]}, indent=2))


if __name__ == "__main__":
    main()
