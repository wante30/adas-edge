"""Stop/restart only the owned AI; verify the same native HMI recovers."""
import json,time
from pathlib import Path
import test_session as session

ROOT=session.ROOT

def main():
    state=json.loads(session.STATE.read_text(encoding="utf-8"))
    assert state["boot_id"]==session.BOOT.read_text().strip()
    label=state["label"];old_pid=state["test_ai_pid"];hmi=state["test_hmi_pid"]
    process=session.record(old_pid)
    assert process["cwd"]==str(ROOT/"ai")
    control_mode=process["env"]["ADAS_CONTROL_MODE"]
    assert control_mode in ("DRY_RUN","HARDWARE")
    metric_path=session.EVIDENCE/(label+"_hmi.jsonl")
    def metric():
        with metric_path.open("rb") as stream:
            stream.seek(0,2);stream.seek(max(0,stream.tell()-12000))
            return json.loads(stream.read().splitlines()[-1])
    def wait(predicate,timeout=60):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            row=metric()
            if predicate(row):return row
            time.sleep(.2)
        raise RuntimeError("Recovery timeout: "+str(metric()))
    def last_result():
        with (session.EVIDENCE/(label+"_ai.jsonl")).open("rb") as stream:
            stream.seek(0,2);stream.seek(max(0,stream.tell()-60000))
            return json.loads(stream.read().splitlines()[-1])
    before=wait(lambda r:r["ai_online"] and r["ai_results"]>20)
    old_bridge=last_result()["inference"]["bridge_id"]
    session.stop(old_pid)
    offline=wait(lambda r:not r["ai_online"] and r["ai_fps"]==0)
    assert offline["produced_frames"]>before["produced_frames"]
    new_pid=session.launch(process["args"],process["cwd"],process["env"],label+"_ai.log")
    state["test_ai_pid"]=new_pid;session.save(state)
    restored=wait(lambda r:r["ai_online"] and r["ai_results"]>before["ai_results"]+20)
    new_bridge=last_result()["inference"]["bridge_id"]
    assert new_bridge!=old_bridge
    assert Path("/proc/"+str(hmi)).exists()
    assert restored["produced_frames"]>offline["produced_frames"]
    report=dict(passed=True,kind="ACTUAL_AI_RESTART",control_mode=control_mode,hmi_pid=hmi,
                old_ai_pid=old_pid,new_ai_pid=new_pid,old_bridge_id=old_bridge,new_bridge_id=new_bridge,
                before=before,offline=offline,restored=restored)
    (session.EVIDENCE/(label+"_restart_acceptance.json")).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)
if __name__=="__main__":main()
