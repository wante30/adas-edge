"""Actual board replay: alternating modes, source/model/Qt FPS, event/UI latency."""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EVIDENCE=ROOT/"evidence"
def rows(path):
    if not path.exists():return []
    out=[]
    with path.open() as f:
        for line in f:
            try:out.append(json.loads(line))
            except ValueError:pass
    return out

def stats(values):
    a=sorted(v for v in values if isinstance(v,(int,float)))
    return dict(count=len(a),mean=sum(a)/len(a) if a else None,p50=a[len(a)//2] if a else None,p95=a[int(.95*(len(a)-1))] if a else None,max=a[-1] if a else None)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--label",required=True);parser.add_argument("--seconds",type=int,default=610);parser.add_argument("--switch-every",type=int,default=60)
    args=parser.parse_args();label=args.label
    metric_path=EVIDENCE/(label+"_hmi.jsonl")
    def metric():
        with metric_path.open('rb') as f:
            f.seek(0,2);f.seek(max(0,f.tell()-12000));return json.loads(f.read().splitlines()[-1])
    def wait(predicate,timeout=90):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            try:
                row=metric()
                if predicate(row):return row
            except (OSError,ValueError,IndexError):pass
            time.sleep(.2)
        raise RuntimeError("Actual HMI timeout: "+str(metric()))
    def click(x,y):subprocess.run(["python3",str(ROOT/"tools/ui_control.py"),str(x),str(y)],check=True,stdout=subprocess.DEVNULL)
    def capture(name):subprocess.run(["xwd","-display",":0","-name","Qt Quick GPU Probe - RK3566","-silent","-out",str(EVIDENCE/(label+"_"+name+".xwd"))],env=dict(os.environ,XAUTHORITY="/home/cat/.Xauthority"),check=True)
    state=json.loads((EVIDENCE/"integration_session.json").read_text());pids=[state[k] for k in ("test_ai_pid","test_hmi_pid")]
    wait(lambda m:m["ai_online"] and m["ai_results"]>20)
    click(390,32);wait(lambda m:m["inference_mode"]=="ROAD_Q1" and m["inference_ready"] and m["ai_online"])
    start=time.monotonic();start_ns=time.monotonic_ns();windows=[];snapshots=[]
    mode="ROAD_Q1";window_start=start_ns;next_switch=start+args.switch_every;next_sample=start
    report=dict(label=label,kind="ACTUAL_1350_FRAME_REPLAY",requested_source_fps=60,control_mode="DRY_RUN",started_ns=start_ns,windows=windows,metrics=snapshots)
    try:
        while time.monotonic()-start<args.seconds:
            now=time.monotonic()
            assert all(Path('/proc/'+str(pid)).exists() for pid in pids),"Owned process exited"
            if now>=next_switch and args.seconds-(now-start)>15:
                end_ns=time.monotonic_ns();windows.append(dict(mode=mode,start_ns=window_start,end_ns=end_ns))
                mode="MOSAIC_FULL" if mode=="ROAD_Q1" else "ROAD_Q1"
                click(480 if mode=="MOSAIC_FULL" else 390,32)
                wait(lambda m:m["inference_mode"]==mode and m["inference_ready"] and m["ai_online"],40)
                window_start=time.monotonic_ns();next_switch+=args.switch_every
                capture(mode+"_"+str(len(windows)))
            if now>=next_sample:
                row=metric();row["sample_ns"]=time.monotonic_ns();snapshots.append(row)
                if len(snapshots)%5==1:print(json.dumps({k:row.get(k) for k in ("inference_mode","model_id","ai_fps","scene_graph_fps","source_fps","ai_results")}),flush=True)
                next_sample+=2
            time.sleep(.15)
        windows.append(dict(mode=mode,start_ns=window_start,end_ns=time.monotonic_ns()));capture("finished")
        report["continuous_seconds"]=time.monotonic()-start
        ai=rows(EVIDENCE/(label+"_ai.jsonl"))
        # Use every emitted 200 ms metric; 2 s polling aliases the render/OCR cadence.
        anchor=snapshots[0]["sample_ns"]-int(snapshots[0]["process_elapsed_ms"]*1000000)
        gui=rows(EVIDENCE/(label+"_hmi.jsonl"))
        for r in gui:r["sample_ns"]=anchor+int(r["process_elapsed_ms"]*1000000)
        perf={}
        for model_mode in ("ROAD_Q1","MOSAIC_FULL"):
            spans=[];selected=[];qt=[];source=[];texture=[]
            for window in windows:
                if window["mode"]!=model_mode:continue
                low=window["start_ns"]+5000000000;high=window["end_ns"]-1000000000
                sample=[r for r in ai if low<=r.get("result_publish_timestamp_ns",0)<=high and r["inference"]["mode"]==model_mode]
                if len(sample)>1:
                    duration=(sample[-1]["result_publish_timestamp_ns"]-sample[0]["result_publish_timestamp_ns"])/1e9
                    spans.append(((len(sample)-1),duration));selected+=sample
                for r in gui:
                    if low<=r["sample_ns"]<=high and r["inference_mode"]==model_mode:
                        qt.append(r["scene_graph_fps"]);source.append(r["source_fps"]);texture.append(r["texture_upload_fps"])
            perf[model_mode]=dict(actual_completions=len(selected),completed_fps=sum(n for n,t in spans)/sum(t for n,t in spans) if spans else 0,
                qt_fps=stats(qt),source_fps=stats(source),texture_fps=stats(texture),
                tracking_ms=stats([r.get("tracking_ms") for r in selected]),risk_fast_ms=stats([r.get("risk_fast_ms") for r in selected]),
                pipeline_ms=stats([r.get("pipeline_ms") for r in selected]),main_rknn_ms=stats([(r.get("main_timing") or {}).get("rknn_call_ms") for r in selected]),
                risk_rules=selected[-1].get("risk_rules") if selected else {},semantics=selected[-1].get("semantics") if selected else {})
        report["performance"]=perf
        all_gui=[r for r in gui if start_ns<=r["sample_ns"]<=windows[-1]["end_ns"]]
        report["qt_scene_graph_fps_mean"]=stats([r["scene_graph_fps"] for r in all_gui])["mean"]
        report["cpu_percent_all_cores"]=stats([r["cpu_percent_of_all_cores"] for r in all_gui])
        report["npu_driver_percent"]=stats([r["npu_load_percent"] for r in all_gui])
        report["gpu_load_percent"]=stats([r["gpu_load_percent"] for r in all_gui])
        report["navigation_states"]={state:sum(r.get("navigation_state")==state for r in ai if start_ns<=r.get("result_publish_timestamp_ns",0)<=windows[-1]["end_ns"]) for state in ("VALID","UNREADABLE","STALE")}
        keys=[(r["inference"]["bridge_id"],r["inference"]["epoch"],r["frame_id"]) for r in ai]
        report["duplicate_result_keys"]=len(keys)-len(set(keys))
        timeline=[r for r in rows(EVIDENCE/(label+"_events.jsonl")) if start_ns<=r.get("logged_ns",0)<=windows[-1]["end_ns"]]
        ui=[r for r in rows(EVIDENCE/(label+"_risk_ui.jsonl")) if start_ns<=r.get("ui_applied_ns",0)<=windows[-1]["end_ns"]]
        confirms={r["event_id"]:r["confirmed_ns"] for r in timeline if r.get("stage")=="event_transition" and r.get("lifecycle_state")=="ACTIVE"}
        first_ui={};last_active={};clear_ui={};unmatched=0;identity_color_mismatches=0
        for row in ui:
            event_map={e["event_id"]:e for e in row.get("events",[])}
            current=set(event_map)
            for box in row.get("marked_boxes",[]):
                event=event_map.get(box["event_id"])
                identity_color_mismatches+=event is None or event.get("severity")!=box.get("severity")
            for event in row.get("events",[]):
                first_ui.setdefault(event["event_id"],row["ui_applied_ns"])
                unmatched+=not event.get("matched",False)
            for eid in last_active.keys()-current:
                clear_ui.setdefault(eid,row["ui_applied_ns"])
            last_active={eid:row["ui_applied_ns"] for eid in current}
        clears={r["event_id"]:r["cleared_ns"] for r in timeline if r.get("lifecycle_state")=="CLEARED"}
        report["event_validation"]=dict(transitions=sum(r.get("stage")=="event_transition" for r in timeline),
            critical_decisions=sum(r.get("stage")=="control_decision" for r in timeline),requests=sum(r.get("stage")=="brake_requested" for r in timeline),
            accepted=sum(r.get("stage")=="brake_accepted" for r in timeline),timeouts=sum(r.get("stage")=="brake_timeout" for r in timeline),
            unmatched_hazard=unmatched,confirm_to_ui_ms=stats([(first_ui[e]-t)/1e6 for e,t in confirms.items() if e in first_ui]),
            clear_to_ui_ms=stats([max(0,(clear_ui[e]-t)/1e6) for e,t in clears.items() if e in clear_ui]))
        report["schedule"]={key:stats([r.get(key) for r in rows(EVIDENCE/(label+"_schedule.jsonl"))]) for key in ("request_spacing_ms","frame_write_send_ms","gui_busy_fraction","backoff_ms")}
        report["scene"]=stats([r.get("scene_fps") for r in rows(EVIDENCE/(label+"_scene.jsonl"))])
        requests=[r for r in timeline if r.get("stage")=="brake_requested"]
        active_ai=[r for r in ai if start_ns<=r.get("result_publish_timestamp_ns",0)<=windows[-1]["end_ns"]]
        report["control_consistency"]=dict(
            identity_color_mismatches=identity_color_mismatches,
            warning_requests=sum(r.get("severity")!="critical" for r in requests),
            non_q1_requests=sum(r.get("source_id")!="Q1" for r in requests),
            cleared_requests=sum(r.get("lifecycle_state")!="ACTIVE" for r in requests),
            missing_critical_decisions=len({e["event_id"] for r in active_ai for e in r.get("hazard_details",[]) if e.get("severity")=="critical"}-{r["event_id"] for r in timeline if r.get("stage")=="control_decision"}))
        report["fps_passed"]=perf["ROAD_Q1"]["completed_fps"]>=6.5 and perf["MOSAIC_FULL"]["completed_fps"]>=13.8 and (report["qt_scene_graph_fps_mean"] or 0)>=28
        report["passed"]=report["fps_passed"] and report["continuous_seconds"]>=600 and not any(report["control_consistency"].values()) and not report["duplicate_result_keys"]
    except BaseException:
        report["error"]=traceback.format_exc();report["passed"]=False;raise
    finally:
        (EVIDENCE/(label+"_acceptance.json")).write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k not in ("metrics","windows")}),flush=True)

if __name__=="__main__":main()
