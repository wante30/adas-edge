"""Replay cached frozen-model outputs with source-time scheduling and fixes."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from tools.adas_hmi_sim.backend.traffic_light_runtime import TrafficLightTemporalFilter
from tools.adas_hmi_sim.backend.traffic_light_specialist import SignalCandidateCache, next_signal_roi
from tools.evaluate_signal_audit import evaluate


def state_metric(rows, label_ranges, state):
    tp=fp=fn=neg=0
    for r in label_ranges:
        wanted=r['vehicle_state']
        if wanted in {'UNKNOWN','UNREVIEWED'}: continue
        for fid in range(r['start_frame'], r['end_frame']+1):
            actual=rows[fid]['traffic_light_state']
            if wanted==state:
                tp+=actual==state; fn+=actual!=state
            else:
                neg+=1; fp+=actual==state
    return {'tp':tp,'fp':fp,'fn':fn,'recall':tp/max(1,tp+fn),
            'precision':tp/max(1,tp+fp) if neg else None,'negative_frames':neg,
            'scope':'REVIEWED_COLOUR_STATE_FRAMES_NOT_DETECTION_AP'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--labels', type=Path)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--main-fps', type=float, default=10.)
    p.add_argument('--specialist-fps', type=float, default=3.)
    p.add_argument('--focused-input', type=Path)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/'frames.csv').exists(): raise SystemExit('Use a fresh output directory')
    summary=json.loads((args.input_dir/'summary.json').read_text(encoding='utf-8'))
    raw={r['frame']:r for r in map(json.loads,(args.input_dir/'raw_predictions.jsonl').open(encoding='utf-8'))}
    focused=({r['frame']:r for r in map(json.loads,(args.focused_input/'raw_predictions.jsonl').open(encoding='utf-8'))}
             if args.focused_input else None)
    next_roi='top60';roi_counts={'top60':0,'focus':0}
    cap=cv2.VideoCapture(summary['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,args.start_frame)
    fps=float(summary['source_fps']);last_main=-1e9;last_spec=-1e9
    modes=['main_only','specialist_only','specialist_fixed','fused_before','fused_fixed']
    filters={n:TrafficLightTemporalFilter(require_lamp_evidence=False,specialist_min_confidence=None) for n in modes}
    filters['fused_fixed']=TrafficLightTemporalFilter(ego_x_max=.95,pedestrian_hold_frames=0)
    filters['specialist_fixed']=TrafficLightTemporalFilter(ego_x_max=.95,pedestrian_hold_frames=0)
    cache=SignalCandidateCache();old_cache=[];md=[]
    rows={n:{} for n in modes};last={};latencies=[];updates={'main':0,'specialist':0}
    writer=cv2.VideoWriter(str(args.output_dir/'before_after.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,540))
    snap={0,30,60,105,360,600,720,1180,1200,1238,1270,1302,1348}
    for fid in range(args.start_frame, len(raw)):
        ok,frame=cap.read()
        if not ok:break
        t=fid/fps;entry=raw[fid];tick=time.perf_counter()
        main_due=t-last_main>=1/args.main_fps-1e-6
        spec_due=t-last_spec>=1/args.specialist_fps-1e-6
        if main_due:
            md=[d for d in entry['main'] if d['confidence']>=.25];last_main=t;updates['main']+=1
        if spec_due:
            old_cache=[d for d in entry['specialist'] if d['confidence']>=.05]
            used_roi=next_roi if focused is not None else 'top60'
            candidate_source=focused[fid] if used_roi=='focus' else entry
            chosen=[d for d in candidate_source['specialist'] if d['confidence']>=.05]
            cache.store(frame,chosen,frame_id=fid,timestamp_ms=t*1000)
            next_roi=next_signal_roi(frame,chosen,used_roi)
            roi_counts[used_roi]+=1
            last_spec=t;updates['specialist']+=1
        signals=cache.current(t*1000)
        for name in modes:
            if main_due or name not in last:
                det={'main_only':md,'specialist_only':old_cache,'specialist_fixed':signals,
                     'fused_before':md+old_cache,'fused_fixed':md+signals}[name]
                e=filters[name].update(frame,det)
                last[name]={'traffic_light_state':e.state,'traffic_light_pedestrian_state':e.pedestrian_state,
                            'decision_state':e.decision_state,'raw_state':e.raw_state}
            rows[name][fid-args.start_frame]=last[name]
        latencies.append((time.perf_counter()-tick)*1000)
        panes=[]
        for name,items in [('fused_before',md+old_cache),('fused_fixed',md+signals)]:
            view=cv2.resize(frame,(960,540));scale=960/frame.shape[1]
            for d in items:
                x1,y1,x2,y2=(round(x*scale) for x in d['bbox'])
                valid=d.get('lamp_evidence_valid',True)
                color=(0,240,0) if valid else (130,130,130)
                cv2.rectangle(view,(x1,y1),(x2,y2),color,1)
            cv2.rectangle(view,(0,488),(960,540),(10,10,10),-1)
            e=last[name];cv2.putText(view,f'{name}  frame={fid}  vehicle={e["traffic_light_state"]}  pedestrian={e["traffic_light_pedestrian_state"]}',
                                  (8,510),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1)
            cv2.putText(view,'MAIN 10 FPS / SPECIALIST 3 FPS | ego permission UNKNOWN',
                        (8,533),cv2.FONT_HERSHEY_SIMPLEX,.43,(190,190,190),1)
            panes.append(view)
        canvas=np.hstack(panes);writer.write(canvas)
        if fid in snap:cv2.imwrite(str(args.output_dir/f'compare_{fid:04d}.jpg'),canvas)
    cap.release();writer.release()
    out={'evidence':'PC_CACHED_MODEL_OUTPUT_REPLAY_NOT_RK3566',
         'source_video':summary['video'],'source_start_frame':args.start_frame,
         'source_fps':fps,'frames':len(rows[modes[0]]),'main_fps':args.main_fps,
         'specialist_fps':args.specialist_fps,'inference_update_counts':updates,
         'specialist_roi_mode':'adaptive' if focused is not None else 'top60','roi_update_counts':roi_counts,
         'postprocess_ms':{'mean':float(np.mean(latencies)),'p95':float(np.percentile(latencies,95))},
         'metric_scope':'VISUALLY_REVIEWED_COLOUR_STATE_FRAMES; NOT DATASET_AP OR EGO_ASSOCIATION',
         'state_counts':{n:{s:sum(r['traffic_light_state']==s for r in rows[n].values()) for s in ['RED','GREEN','YELLOW','UNKNOWN']} for n in modes}}
    if args.labels:
        labels=json.loads(args.labels.read_text(encoding='utf-8'))
        out['labels']=str(args.labels.resolve());out['metrics']={n:evaluate(rows[n],labels) for n in modes}
        out['vehicle_red']={n:state_metric(rows[n],labels['ranges'],'RED') for n in modes}
    with (args.output_dir/'frames.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['frame']+[f'{n}_{key}' for n in modes for key in last[n]])
        writer.writeheader()
        for fid in rows[modes[0]]:
            writer.writerow({'frame':fid,**{f'{n}_{key}':value for n in modes for key,value in rows[n][fid].items()}})
    (args.output_dir/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'frames':out['frames'],'updates':updates,'postprocess_ms':out['postprocess_ms'],
                      'state_counts':out['state_counts'],
                      'metrics':{n:{k:{key:metric[key] for key in ['tp','fp','fn','recall']}
                                    for k,metric in v.items() if k in ['vehicle_green','pedestrian_red']}
                                 for n,v in out.get('metrics',{}).items()},
                      'vehicle_red':out.get('vehicle_red')},indent=2))


if __name__=='__main__':main()
