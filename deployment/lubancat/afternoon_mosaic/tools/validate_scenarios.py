"""Synthetic rare-scene validation, explicitly separate from actual video evidence."""
import contextlib,io,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"ai"))
from light_risk import LightRiskEngine
from risk_events import EventStore,RiskObservation
from auto_brake import AutoBrakeController
from control_scenarios import crosswalk_pulse_candidate
class Audit:
    def __init__(self):self.rows=[]
    def write(self,stage,**row):self.rows.append(dict(stage=stage,synthetic_validation=True,**row))
def main():
    os.environ["ADAS_CONTROL_MODE"]="DRY_RUN"
    audit=Audit();results={};context=dict(mode="ROAD_Q1",epoch="synthetic-1",bridge_id="SYNTHETIC")
    cases=[("stationary_large","car",False,False,False),("closing","car",True,False,False),
        ("pedestrian_crossing","person",False,True,False),("bicycle_entry","bicycle",False,True,False),
        ("motorcycle_entry","motorcycle",False,True,False),("vehicle_cut_in","car",False,True,False),
        ("oncoming_unverified","car",True,False,False),("oncoming_verified","car",True,False,True),
        ("braking_closing","car",True,False,False),("crosswalk_person","person",False,True,False)]
    for case,cls,growing,lateral,direction_verified in cases:
        engine=LightRiskEngine();store=EventStore(audit);control=AutoBrakeController(enabled=True,journal=audit)
        seen=[];active=();start_rows=len(audit.rows)
        for i in range(22):
            ts=1000000000+i*100000000;now=ts+20000000
            h=min(.65,.18+i*.035) if growing else .35
            x=max(.5,.85-i*.03) if lateral else .5;w=.13;foot=.9
            d=dict(bbox=[(x-w/2)*640,(foot-h)*360,(x+w/2)*640,foot*360],class_name=cls,source_id="Q1",track_id=1,confidence=.9,synthetic=True)
            if case.startswith("oncoming"):d.update(direction="oncoming",direction_verified=direction_verified)
            semantics={}
            if case=="braking_closing":semantics=dict(lamps={"1":dict(rise=True,timestamp_ns=ts)})
            if case=="crosswalk_person":semantics=dict(crosswalks=[dict(bbox=[.2,.6,.8,.95],confidence=.9)])
            candidates=engine.update([d] if i<18 else [],ts,("ROAD_Q1",case,"MTL"),semantics)
            active=store.update(candidates,**context,model_id="SYNTHETIC",frame_id=i,timestamp_ns=ts,now_ns=now)
            seen+=list(active)
            control.update(RiskObservation(events=active),context=context,now_ns=now)
        transitions=audit.rows[start_rows:]
        assert not active,"Synthetic sequence failed to clear"
        requests=[r for r in transitions if r["stage"]=="brake_requested"]
        if case=="stationary_large":assert not requests
        if case=="oncoming_unverified":assert not any(e["kind"]=="ONCOMING_PATH_CONFLICT" and e["severity"]=="critical" for e in seen)
        results[case]=dict(events=sorted({e["kind"]+":"+e["severity"] for e in seen}),requests=len(requests),cleared=True,rule_metrics=engine.metrics())
        if case=="crosswalk_person":
            crossings=[e for e in seen if e["kind"]=="PEDESTRIAN_CROSSING_PATH" and e["severity"]=="critical" and e["evidence"].startswith("MTL_CROSSWALK") and e.get("control_eligible")]
            assert crossings,"Crosswalk crossing critical evidence missing"
            e=crosswalk_pulse_candidate(crossings[-1]);now=e["timestamp_ns"]+20000000
            control.update(RiskObservation(events=[e]),context=context,now_ns=now)
            control.update(RiskObservation(events=[e]),context=context,now_ns=now+350000000)
            control.update(RiskObservation(events=[]),context=context,now_ns=now+450000000)
        control.close()
    required={"FRONT_COLLISION_RISK","PEDESTRIAN_IN_PATH","PEDESTRIAN_CROSSING_PATH","TWO_WHEELER_PATH_CONFLICT","VEHICLE_CUT_IN","ONCOMING_PATH_CONFLICT","SUSPECTED_BRAKING_WITH_CLOSING"}
    actual={s.split(":")[0] for case in results.values() for s in case["events"] if s.endswith(":critical")}
    assert required<=actual,str(required-actual)
    assert any(r["stage"]=="pulse_ended" for r in audit.rows)
    out=ROOT/"validation/followup";out.mkdir(parents=True,exist_ok=True)
    (out/"synthetic_scenarios.json").write_text(json.dumps(dict(kind="SYNTHETIC_NUMERICAL_SEQUENCES",actual_video=False,hardware_executed=False,passed=True,cases=results),indent=2),encoding="utf-8")
    (out/"synthetic_control_timeline.jsonl").write_text("".join(json.dumps(r,separators=(",",":"))+"\n" for r in audit.rows),encoding="utf-8")
    print("SYNTHETIC_SCENARIOS_PASSED",len(results))
if __name__=="__main__":
    with contextlib.redirect_stdout(io.StringIO()):main()
    print("SYNTHETIC_SCENARIOS_PASSED")
