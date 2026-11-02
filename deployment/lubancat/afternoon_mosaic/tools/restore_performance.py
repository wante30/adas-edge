"""Restore the preserved PR7 Q4 runtime without enabling physical actuation."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
import test_session as session
p=argparse.ArgumentParser();p.add_argument("--baseline-root",type=Path,default=Path("/home/cat/adas_risk_dev_20261102"));args=p.parse_args()
baseline=args.baseline_root.resolve();assert (baseline/"build-stage4/qt_gpu_probe").is_file()
state=json.loads(session.STATE.read_text());assert state["boot_id"]==session.BOOT.read_text().strip()
for key in ("test_ai_pid","test_hmi_pid"):
 pid=state.get(key)
 if pid and Path(f"/proc/{pid}/cmdline").exists():
  cmd=Path(f"/proc/{pid}/cmdline").read_bytes()
  assert (b"isolated_ai.py" in cmd if key=="test_ai_pid" else b"qt_gpu_probe" in cmd)
  session.stop(pid)
(baseline/"evidence").mkdir(exist_ok=True)
(baseline/"evidence/integration_session.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
subprocess.run(["python3",str(baseline/"tools/test_session.py"),"start","--build","build-stage4","--label","restored_a9054ec2","--ai-hz","0","--control-mode","HARDWARE","--external-qml"],check=True,
 env=dict(__import__("os").environ,ADAS_CONTROL_AUTHORIZED="0"))
print("Restored preserved a9054ec2 executable and AI; Q4 forward; real telemetry; actuator sends OFF")
