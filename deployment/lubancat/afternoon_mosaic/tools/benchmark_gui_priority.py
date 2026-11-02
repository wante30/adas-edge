"""Bounded reversible real PCIe scheduler trial; no control sends."""
import argparse,json,os,subprocess,time
from pathlib import Path
import profile_system as profile
from validate_performance import select
p=argparse.ArgumentParser();p.add_argument("--label",required=True);p.add_argument("--seconds",type=int,default=60);p.add_argument("--model",default="MTL-FAST");p.add_argument("--ai-nice",type=int);a=p.parse_args()
state=json.loads((profile.EVIDENCE/"integration_session.json").read_text())
assert not state["control_authorized"] and state["input_kind"]=="REAL_PCIE"
select(state["label"],a.model);time.sleep(4)
pid=state["test_hmi_pid"];tasks={}
for path in Path("/proc/%d/task"%pid).iterdir():
    name=(path/"comm").read_text().strip()
    if int(path.name)==pid or name in ("QSGRenderThread","mali-cmar-backe"):
        tasks[int(path.name)]=os.getpriority(os.PRIO_PROCESS,int(path.name))
ai_tid=None;ai_old=None
if a.ai_nice is not None:
    # Diagnostic only: choose the recorded dominant inference thread, never the OCR process.
    candidates=[]
    for task in Path("/proc/%d/task"%state["test_ai_pid"]).iterdir():
        fields=(task/"stat").read_text().split(") ")[1].split()
        if int(task.name)!=state["test_ai_pid"]:candidates.append((int(fields[11])+int(fields[12]),int(task.name)))
    ai_tid=max(candidates)[1];ai_old=os.getpriority(os.PRIO_PROCESS,ai_tid)
report=dict(pid=pid,tasks=tasks,requested_nice=-5,model=a.model,ai_tid=ai_tid,ai_old=ai_old,
    ai_nice=a.ai_nice,started_ns=time.monotonic_ns())
try:
    for tid in tasks:subprocess.run(["sudo","-n","renice","-n","-5","-p",str(tid)],check=True)
    if ai_tid:subprocess.run(["sudo","-n","renice","-n",str(a.ai_nice),"-p",str(ai_tid)],check=True)
    time.sleep(4);profile.sample(a.label,a.seconds)
    observed=set()
    for line in (profile.EVIDENCE/(a.label+"_system.jsonl")).read_text().splitlines():
        for row in json.loads(line).get("ai",[]):observed.add(row.get("model_id"))
    report["observed_models"]=sorted(observed)
    report["valid_single_model"]=observed=={a.model}
finally:
    for tid,nice in tasks.items():
        subprocess.run(["sudo","-n","renice","-n",str(nice),"-p",str(tid)],check=True)
    if ai_tid:subprocess.run(["sudo","-n","renice","-n",str(ai_old),"-p",str(ai_tid)],check=True)
    report.update(restored=True,ended_ns=time.monotonic_ns())
    (profile.EVIDENCE/(a.label+"_scheduler_trial.json")).write_text(json.dumps(report,indent=2),encoding="utf-8")
