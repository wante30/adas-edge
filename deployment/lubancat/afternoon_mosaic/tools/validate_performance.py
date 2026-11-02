"""Real Qt menu/model/PCIe acceptance; hardware sends must remain unauthorized."""
import argparse,json,os,subprocess,time
from pathlib import Path
from validate_followup import rows
ROOT=Path(__file__).resolve().parents[1];EV=ROOT/"evidence"

def last(path):
    with path.open("rb") as f:
        f.seek(0,2);f.seek(max(0,f.tell()-20000));lines=f.read().splitlines()
    for line in reversed(lines):
        try:return json.loads(line)
        except ValueError:pass
    raise RuntimeError("No complete metric row")

def click(x,y):
    subprocess.run(["python3",str(ROOT/"tools/ui_control.py"),str(x),str(y)],check=True,stdout=subprocess.DEVNULL)

def capture(label,name):
    subprocess.run(["xwd","-display",":0","-name","Qt Quick GPU Probe - RK3566","-silent","-out",str(EV/(label+"_"+name+".xwd"))],env=dict(os.environ,XAUTHORITY="/home/cat/.Xauthority"),check=True)

def select(label,model,timeout=60):
    path=EV/(label+"_hmi.jsonl");before=last(path)["ai_results"]
    y={"MTL-FAST":304,"FAST-Q4":348,"FAST-MOSAIC":392}[model]
    click(1200,40);time.sleep(.4);click(1000,y)
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        r=last(path)
        if r["model_id"]==model and r["inference_ready"] and r["ai_online"] and r["ai_results"]>=before+15:return r
        time.sleep(.2)
    raise RuntimeError("Model APPLIED/fresh result timeout: "+json.dumps(last(path)))

def main():
    p=argparse.ArgumentParser();p.add_argument("--label",required=True);p.add_argument("--seconds",type=int,default=610);p.add_argument("--window",type=int,default=100);args=p.parse_args()
    label=args.label;state=json.loads((EV/"integration_session.json").read_text())
    assert state["label"]==label and state["input_kind"]=="REAL_PCIE" and state["control_mode"]=="HARDWARE" and state["forward_quadrant"]==3
    assert b"ADAS_CONTROL_AUTHORIZED=0" in Path(f"/proc/{state['test_ai_pid']}/environ").read_bytes().split(b"\0")
    assert b"--raw-replay" not in Path(f"/proc/{state['test_hmi_pid']}/cmdline").read_bytes()
    deadline=time.monotonic()+90
    r={}
    while time.monotonic()<deadline:
        try:r=last(EV/(label+"_hmi.jsonl"))
        except (FileNotFoundError,RuntimeError):
            time.sleep(.2)
            continue
        if r["ai_online"] and r["ai_results"]>20:break
        time.sleep(.2)
    assert r.get("ai_online"), "AI did not become ready"
    select(label,"MTL-FAST")
    start=time.monotonic_ns();report=dict(kind="REAL_PCIE_FULL_SYSTEM",started_ns=start,seconds=args.seconds,control_authorized=False,windows=[],switches=[])
    sampler=subprocess.Popen(["python3",str(ROOT/"tools/profile_system.py"),"--label",label,"--seconds",str(args.seconds)])
    models=("MTL-FAST","FAST-Q4","FAST-MOSAIC");index=0
    try:
        while (time.monotonic_ns()-start)/1e9<args.seconds:
            model=models[index%3];window_start=time.monotonic_ns()
            if index:
                applied=select(label,model)
                report["switches"].append(dict(timestamp_ns=time.monotonic_ns(),model=model,hmi=applied))
            capture(label,model+"_"+str(index))
            if model=="FAST-Q4" and index==1:
                click(1200,40);time.sleep(.5);capture(label,"FAST-Q4_menu");click(1229,78)
            end=min(start+int(args.seconds*1e9),start+int((index+1)*args.window*1e9))
            while time.monotonic_ns()<end:time.sleep(.5)
            report["windows"].append(dict(model=model,start_ns=window_start,end_ns=time.monotonic_ns()))
            (EV/(label+"_performance_acceptance.json")).write_text(json.dumps(report,indent=2),encoding="utf-8")
            index+=1
    except BaseException as exc:
        sampler.terminate()
        sampler.wait(timeout=10)
        report["failure"]=str(exc)
        report["passed_functional_checks"]=False
        (EV/(label+"_performance_acceptance.json")).write_text(json.dumps(report,indent=2),encoding="utf-8")
        raise
    else:
        sampler.wait(timeout=20)
        assert sampler.returncode==0
        report["ended_ns"]=time.monotonic_ns();report["elapsed_s"]=(report["ended_ns"]-start)/1e9
        ai=[r for r in rows(EV/(label+"_ai.jsonl")) if start<=r.get("result_publish_timestamp_ns",0)<=report["ended_ns"]]
        events=[r for r in rows(EV/(label+"_events.jsonl")) if start<=r.get("logged_ns",0)<=report["ended_ns"]]
        requests=[r for r in events if r.get("stage") in ("control_request","brake_requested")]
        assert not requests,"Actuation is forbidden during performance acceptance"
        report["real_control_requests"]=len(requests)
        report["fast_q4_stale_mtl_rows"]=sum(bool((r.get("semantics") or {}).get("mtl_available")) for r in ai if r["model_id"]=="FAST-Q4")
        report["wrong_forward_rows"]=sum(r.get("forward_source_id")!="Q4" for r in ai)
        ui=[r for r in rows(EV/(label+"_risk_ui.jsonl")) if start<=r.get("ui_applied_ns",0)<=report["ended_ns"]]
        mismatch=0
        for r in ui:
            eventmap={e["event_id"]:e for e in r.get("events",[])}
            for b in r.get("marked_boxes",[]):
                e=eventmap.get(b["event_id"])
                mismatch+=not e or e["severity"]!=b["severity"]
        report["box_event_mismatches"]=mismatch
        report["result_rows"]=len(ai);report["ui_samples"]=len(ui)
        report["passed_functional_checks"]=not (report["fast_q4_stale_mtl_rows"] or report["wrong_forward_rows"] or mismatch)
        (EV/(label+"_performance_acceptance.json")).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)
if __name__=="__main__":main()
