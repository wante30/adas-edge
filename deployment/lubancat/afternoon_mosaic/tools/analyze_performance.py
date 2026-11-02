"""Summarize fixed monotonic windows without confusing process/system CPU."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import statistics


def dist(values):
    v=sorted(float(x) for x in values if x is not None)
    if not v:return {}
    def q(p):return v[min(len(v)-1,int((len(v)-1)*p))]
    return dict(n=len(v),mean=statistics.mean(v),p10=q(.1),p50=q(.5),p95=q(.95),max=v[-1])


def analyze(path):
    lines=[json.loads(l) for l in Path(path).read_text().splitlines()]
    return analyze_rows(lines, str(path))


def analyze_rows(lines, label="samples", contiguous=True):
    rows=[r for r in lines if r.get("kind")=="sample"]
    hmi=[h for r in rows for h in r["hmi"]]
    ai=[a for r in rows for a in r["ai"]]
    groups=defaultdict(list)
    for a in ai:
        i=a.get("inference",{})
        groups[(i.get("bridge_id"),str(i.get("epoch")),a.get("model_id"))].append(a)
    rates=defaultdict(lambda:dict(intervals=0,seconds=0,results=0))
    for key,values in groups.items():
        t=sorted(set(a["result_publish_timestamp_ns"] for a in values))
        if len(t)>1:
            r=rates[key[-1]];r["intervals"]+=len(t)-1;r["seconds"]+=(t[-1]-t[0])/1e9;r["results"]+=len(t)
    for r in rates.values():r["fps"]=r["intervals"]/r["seconds"]
    dt=sum(r["dt_s"] for r in rows)
    proc=defaultdict(lambda:dict(cpu_seconds=0,run_ms=0,runqueue_wait_ms=0))
    thread=defaultdict(lambda:dict(cpu_seconds=0,run_ms=0,runqueue_wait_ms=0))
    for r in rows:
        for t in r["threads"]:
            for dest,key in ((proc,str(t["pid"])+" "+t["cmd"][:150]),(thread,str(t["pid"])+":"+str(t["tid"])+" "+t["name"])):
                d=dest[key];d["cpu_seconds"]+=t["cpu_one_core_percent"]/100*r["dt_s"]
                d["run_ms"]+=t["run_ms"];d["runqueue_wait_ms"]+=t["runqueue_wait_ms"]
    for data in (proc,thread):
        for r in data.values():r["cpu_one_core_percent"]=r["cpu_seconds"]/dt*100
    stages={k:dist(a.get(k) for a in ai) for k in ("decode_ms","pipeline_ms","tracking_ms","risk_fast_ms","packet_encode_ms","shared_frame_read_ms","shared_frame_validate_ms","analysis_thread_cpu_ms","decode_thread_cpu_ms","pipeline_thread_cpu_ms","tracking_thread_cpu_ms")}
    for k in ("preprocess_ms","rknn_call_ms","decode_nms_ms","seg_ms","overlay_encode_ms"):
        stages[k]=dist((a.get("main_timing") or {}).get(k) for a in ai)
    result=dict(file=label,metadata=lines[0],seconds=dt,system_cpu_percent=dist(r["system_cpu_percent"] for r in rows),
        hmi={k:dist(h.get(k) for h in hmi) for k in ("scene_graph_fps","texture_upload_fps","source_fps","ai_gui_update_ms","ai_result_age_ms","gpu_load_percent","npu_load_percent","pgv_read_ms","pgv_crc_ms","texture_upload_ms","source_to_upload_ms")},
        ai=rates, stages_ms=stages,
        result_age_ms=dist((a["result_publish_timestamp_ns"]-a["capture_timestamp_ns"])/1e6 for a in ai),
        process_cpu=sorted(proc.items(),key=lambda x:-x[1]["cpu_seconds"])[:12],
        thread_cpu=sorted(thread.items(),key=lambda x:-x[1]["cpu_seconds"])[:20],
        navigation=dict(Counter(a.get("navigation_state") for a in ai)),
        lvt1_online=sum(bool(r.get("lvt1",{}).get("online")) for r in rows),samples=len(rows),
        capture_running=sum(r.get("capture",{}).get("state")=="running" for r in rows))
    if len(hmi)>1:
        # Metrics intervals vary when GUI is busy. Weight rates by their actual
        # process-monotonic interval, including stalls rather than rewarding them.
        weighted={k:[0.,0.] for k in ("scene_graph_fps","texture_upload_fps","source_fps")}
        for before,after in zip(hmi,hmi[1:]):
            elapsed=(after["process_elapsed_ms"]-before["process_elapsed_ms"])/1000
            if elapsed<=0 or (not contiguous and elapsed>2):continue
            for key,value in weighted.items():
                if after.get(key) is not None:value[0]+=after[key]*elapsed;value[1]+=elapsed
        result["hmi_time_weighted_fps"]={k:v[0]/v[1] for k,v in weighted.items() if v[1]}
    if contiguous and len(rows)>1:
        a,b=rows[0],rows[-1]
        seconds=(b["timestamp_ns"]-a["timestamp_ns"])/1e9
        captures=[r["capture"] for r in rows if r.get("capture",{}).get("state")=="running"]
        if len(captures)>1:
            result["capture_counter_fps"]=(captures[-1]["frames"]-captures[0]["frames"])/((captures[-1]["updated_ns"]-captures[0]["updated_ns"])/1e9)
    if contiguous and len(hmi)>1:
        elapsed=(hmi[-1]["process_elapsed_ms"]-hmi[0]["process_elapsed_ms"])/1000
        result["source_counter_fps"]=(hmi[-1]["produced_frames"]-hmi[0]["produced_frames"])/elapsed
        result["texture_counter_fps"]=(hmi[-1]["uploaded_frames"]-hmi[0]["uploaded_frames"])/elapsed
        result["fpga_frame_id_rate"]=(hmi[-1]["source_frame_id"]-hmi[0]["source_frame_id"])/elapsed
        result["gui_long_updates_delta"]=hmi[-1].get("gui_long_update_count",0)-hmi[0].get("gui_long_update_count",0)
        result["gui_long_update_ms_delta"]=hmi[-1].get("gui_long_update_total_ms",0)-hmi[0].get("gui_long_update_total_ms",0)
        result["source_dropped_delta"]=hmi[-1]["source_dropped_frames"]-hmi[0]["source_dropped_frames"]
        result["crc_errors_delta"]=hmi[-1]["source_crc_errors"]-hmi[0]["source_crc_errors"]
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("files",nargs="+");p.add_argument("--output");args=p.parse_args()
    results=[analyze(f) for f in args.files]
    if args.output:Path(args.output).write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
    for r in results:
        print(Path(r["file"]).name,"CPU",round(r["system_cpu_percent"]["mean"],2),
              {k:round(v.get("mean",0),2) for k,v in r["hmi"].items()},dict(r["ai"]))
        print("TOP",[(k,round(v["cpu_one_core_percent"],1)) for k,v in r["thread_cpu"][:6]])
