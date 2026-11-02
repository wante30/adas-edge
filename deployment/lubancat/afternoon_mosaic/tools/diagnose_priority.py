"""Bounded A/B of GUI/render scheduling; restores original nice values."""
import json,os,subprocess,time
from pathlib import Path
import profile_system as profile
from validate_performance import select
root=profile.ROOT;state=json.loads((profile.EVIDENCE/"integration_session.json").read_text());pid=state["test_hmi_pid"]
assert b"qt_gpu_probe" in Path(f"/proc/{pid}/cmdline").read_bytes()
select(state["label"],"MTL-FAST");time.sleep(10)
profile.sample("priority_zero",35)
tasks={}
for p in Path(f"/proc/{pid}/task").iterdir():
 name=(p/"comm").read_text().strip()
 if p.name==str(pid) or name in ("QSGRenderThread","mali-cmar-backe"):
  tasks[int(p.name)]=os.getpriority(os.PRIO_PROCESS,int(p.name))
try:
 for tid in tasks:subprocess.run(["sudo","-n","renice","-n","-5","-p",str(tid)],check=True)
 time.sleep(5);profile.sample("priority_minus5",35)
finally:
 for tid,nice in tasks.items():subprocess.run(["sudo","-n","renice","-n",str(nice),"-p",str(tid)],check=True)
(profile.EVIDENCE/"priority_trial.json").write_text(json.dumps(dict(pid=pid,tasks=tasks,restored=True),indent=2))
