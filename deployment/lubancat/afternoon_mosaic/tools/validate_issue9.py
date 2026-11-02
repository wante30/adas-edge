"""Real PCIe, real menu, bounded no-send workload/performance acceptance."""
import argparse,json,os,subprocess,time
from pathlib import Path
from validate_performance import select,capture,last
from profile_system import sample
ROOT=Path(__file__).resolve().parents[1];EV=ROOT/"evidence"

def main():
    p=argparse.ArgumentParser();p.add_argument("--label",required=True);p.add_argument("--seconds",type=int,default=60)
    p.add_argument("--models",nargs="+",default=["FAST-MOSAIC","FAST-MOSAIC-LITE","MTL-FAST","FAST-Q4"])
    a=p.parse_args();state=json.loads((EV/"integration_session.json").read_text())
    assert state["label"]==a.label and state["input_kind"]=="REAL_PCIE" and not state["control_authorized"]
    assert state["forward_quadrant"]==3
    report=dict(kind="ISSUE9_REAL_PCIE_NO_SEND",windows=[],started_ns=time.monotonic_ns())
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        try:
            ready=last(EV/(a.label+"_hmi.jsonl"))
            if ready.get("ai_online") and ready.get("ai_results",0)>20:break
        except (FileNotFoundError,RuntimeError):pass
        time.sleep(.2)
    else:raise RuntimeError("AI startup timeout")
    for index,model in enumerate(a.models):
        applied=select(a.label,model)
        time.sleep(4)
        start=time.monotonic_ns()
        capture(a.label,model+"_"+str(index))
        sample(a.label,a.seconds)
        # profile_system reuses the label, preserve independent windows.
        profile=EV/(a.label+"_system.jsonl")
        saved=EV/(a.label+"_"+str(index)+"_system.jsonl")
        profile.replace(saved)
        now=time.monotonic_ns()
        rows=[]
        with (EV/(a.label+"_ai.jsonl")).open() as stream:
            for line in stream:
                try:r=json.loads(line)
                except ValueError:continue
                if start<=r.get("result_publish_timestamp_ns",0)<=now:rows.append(r)
        assert len(rows)>10 and all(r["model_id"]==model for r in rows)
        assert all(not (r.get("control") or {}).get("authorized") for r in rows)
        if model=="FAST-MOSAIC-LITE":
            assert all(r.get("workload",{}).get("lite") and r["workload"].get("ocr_pid") is None
                and not r["workload"].get("semantics_active") and not (r.get("control") or {}).get("enabled") for r in rows)
        else:
            assert all(r.get("workload",{}).get("semantics_active") and r["workload"].get("ocr_pid") for r in rows)
        report["windows"].append(dict(model=model,start_ns=start,end_ns=now,applied=applied,
            profile=str(saved),ai_results=len(rows),workload=rows[-1].get("workload"),control=rows[-1].get("control")))
        (EV/(a.label+"_issue9_acceptance.json")).write_text(json.dumps(report,indent=2),encoding="utf-8")
    events=[]
    path=EV/(a.label+"_events.jsonl")
    if path.exists():
        for line in path.read_text().splitlines():
            try:r=json.loads(line)
            except ValueError:continue
            if r.get("logged_ns",0)>=report["started_ns"]:events.append(r)
    assert not [r for r in events if r.get("stage")=="brake_requested"]
    report.update(passed=True,ended_ns=time.monotonic_ns())
    (EV/(a.label+"_issue9_acceptance.json")).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)
if __name__=="__main__":main()
