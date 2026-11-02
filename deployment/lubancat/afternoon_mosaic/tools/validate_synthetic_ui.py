"""Capture real Qt rendering of explicitly labelled synthetic event transitions."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EVIDENCE=ROOT/"evidence";LABEL=os.environ.get("ADAS_SYNTHETIC_LABEL","synthetic_ui")
def tail(path):
    if not path.exists():return []
    with path.open("rb") as f:
        f.seek(0,2);f.seek(max(0,f.tell()-60000));data=f.read().splitlines()
    out=[]
    for line in data:
        try:out.append(json.loads(line))
        except ValueError:pass
    return out

def main():
    subprocess.run(["python3",str(ROOT/"tools/test_session.py"),"start","--build","build-stage4","--label",LABEL,"--ai-hz","0","--control-mode","DRY_RUN","--external-qml","--synthetic-ui","--raw-replay","/home/cat/qt_gpu_probe_ai_20261019/local_quad_full.rgb"],check=True)
    screenshots=[]
    for stage in ("warning","critical","pedestrian","pulse","cleared"):
        deadline=time.monotonic()+40
        while time.monotonic()<deadline:
            timeline=tail(EVIDENCE/(LABEL+"_events.jsonl"))
            candidates=[r for r in timeline if r.get("stage")=="synthetic_ui_stage" and r.get("stage_name")==stage]
            if candidates:
                event=candidates[-1];events=event["events"]
                wanted={e["event_id"] for e in events}
                applied=tail(EVIDENCE/(LABEL+"_risk_ui.jsonl"))
                if applied and {e["event_id"] for e in applied[-1].get("events",[])}==wanted:
                    active=applied[-1].get("events",[])
                    if stage!="cleared" and not active:time.sleep(.1);continue
                    if stage=="cleared" and active:time.sleep(.1);continue
                    if stage=="critical" and not any(e["severity"]=="critical" for e in active):time.sleep(.1);continue
                    if stage in ("pedestrian","pulse") and not any(e["target_class"]=="person" for e in active):time.sleep(.1);continue
                    time.sleep(1.6 if stage=="warning" else .4)
                    path=EVIDENCE/(LABEL+"_"+stage+".xwd")
                    subprocess.run(["xwd","-display",":0","-name","Qt Quick GPU Probe - RK3566","-silent","-out",str(path)],env=dict(os.environ,XAUTHORITY="/home/cat/.Xauthority"),check=True)
                    screenshots.append(dict(stage=stage,path=str(path),ui=applied[-1],control=event["control"]));print(stage,flush=True);break
            time.sleep(.1)
        else:raise RuntimeError("Synthetic UI stage missing: "+stage)
    timeline=[json.loads(l) for l in (EVIDENCE/(LABEL+"_events.jsonl")).read_text().splitlines()]
    assert any(r["stage"]=="brake_accepted" for r in timeline)
    assert any(r["stage"]=="pulse_simulated_execution" for r in timeline)
    assert any(r["stage"]=="pulse_ended" for r in timeline)
    assert not screenshots[-1]["ui"]["events"]
    report=dict(kind="SYNTHETIC_NATIVE_QT_UI_TEST",actual_video_detection=False,passed=True,screenshots=screenshots)
    (EVIDENCE/"synthetic_ui_acceptance.json").write_text(json.dumps(report,indent=2));print("SYNTHETIC_UI_PASSED",flush=True)
if __name__=="__main__":main()
