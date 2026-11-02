"""Additional evidence from the completed same-input board regression."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
from validate_followup import rows,stats

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];ev=root/'evidence';label=args.label
    report=json.loads((ev/(label+'_acceptance.json')).read_text(encoding='utf-8'))
    lo=report['started_ns'];hi=report['windows'][-1]['end_ns']
    ai=[r for r in rows(ev/(label+'_ai.jsonl')) if lo<=r['result_publish_timestamp_ns']<=hi]
    qt=rows(ev/(label+'_qt_ai.jsonl'))
    timing_keys=('decode_ms','shared_frame_read_ms','pipeline_ms','tracking_ms','risk_fast_ms','navigation_submit_ms','box_mapping_ms','status_build_ms','packet_encode_ms','result_publish_ms')
    details={mode:{k:stats([r.get(k) for r in ai if r['inference']['mode']==mode]) for k in timing_keys} for mode in ('ROAD_Q1','MOSAIC_FULL')}
    for mode in details:
        selected=[r for r in ai if r['inference']['mode']==mode]
        details[mode]['rknn_ms']=stats([r['main_timing'].get('rknn_call_ms') for r in selected])
        details[mode]['capture_to_publish_ms']=stats([(r['result_publish_timestamp_ns']-r['capture_timestamp_ns'])/1e6 for r in selected])
    windows=[]
    for w in report['windows']:
        selected=[r for r in ai if w['start_ns']+5000000000<=r['result_publish_timestamp_ns']<=w['end_ns']-1000000000 and r['inference']['mode']==w['mode']]
        if len(selected)>1:windows.append(dict(mode=w['mode'],full_window=(w['end_ns']-w['start_ns'])>=45000000000,results=len(selected),fps=(len(selected)-1)*1e9/(selected[-1]['result_publish_timestamp_ns']-selected[0]['result_publish_timestamp_ns'])))
    decay={}
    for mode in ('ROAD_Q1','MOSAIC_FULL'):
        selected=[w['fps'] for w in windows if w['mode']==mode and w['full_window']]
        early=sum(selected[:2])/len(selected[:2]);late=sum(selected[-2:])/len(selected[-2:])
        decay[mode]=dict(first_two_windows_mean=early,last_two_windows_mean=late,change_percent=100*(late/early-1))
    # PR #6 method: every completion in each contiguous bridge/epoch/model run.
    groups={}
    for row in ai:
        identity=row['inference'];key=(identity['bridge_id'],identity['epoch'],row['model_id'],identity['mode'])
        groups.setdefault(key,[]).append(row)
    baseline_rates={}
    for mode in ('ROAD_Q1','MOSAIC_FULL'):
        spans=[v for k,v in groups.items() if k[3]==mode and len(v)>1]
        n=sum(len(v)-1 for v in spans)
        seconds=sum((v[-1]['result_publish_timestamp_ns']-v[0]['result_publish_timestamp_ns'])/1e9 for v in spans)
        baseline_rates[mode]=dict(completed_fps=n/seconds,results=sum(len(v) for v in spans),measured_seconds=seconds)
    first,last=report['metrics'][0],report['metrics'][-1]
    counter_seconds=(last['process_elapsed_ms']-first['process_elapsed_ms'])/1000
    source_fps=(last['produced_frames']-first['produced_frames'])/counter_seconds
    texture_fps=(last['uploaded_frames']-first['uploaded_frames'])/counter_seconds
    errors=[]
    for suffix in ('_ai.log','_hmi.log'):
        for line in (ev/(label+suffix)).read_text(encoding='utf-8',errors='replace').splitlines():
            if any(marker in line for marker in ('INFER_ERROR','REQUEST_LISTENER_ERROR','double free','corrupted','Segmentation fault','Aborted','Traceback','GL error at')):errors.append(line)
    timeline=[r for r in rows(ev/(label+'_events.jsonl')) if lo<=r.get('logged_ns',0)<=hi]
    ui=[r for r in rows(ev/(label+'_risk_ui.jsonl')) if lo<=r.get('ui_applied_ns',0)<=hi]
    confirms={r['event_id']:r['confirmed_ns'] for r in timeline if r.get('stage')=='event_transition' and r.get('lifecycle_state')=='ACTIVE' and r.get('confirmed_ns',0)>=lo}
    clears={r['event_id']:r['cleared_ns'] for r in timeline if r.get('stage')=='event_transition' and r.get('lifecycle_state')=='CLEARED'}
    first_ui={};clear_ui={};previous=set();mismatches=0;unmatched=0
    for row in ui:
        events={e['event_id']:e for e in row.get('events',[])}
        for eid,event in events.items():
            first_ui.setdefault(eid,row['ui_applied_ns']);unmatched+=not event.get('matched',False)
        for eid in previous-events.keys():clear_ui.setdefault(eid,row['ui_applied_ns'])
        for b in row.get('marked_boxes',[]):
            e=events.get(b['event_id']);mismatches+=e is None or b['severity']!=e['severity']
        previous=set(events)
    requests=[r for r in timeline if r.get('stage')=='brake_requested']
    control_consistency=dict(identity_color_mismatches=mismatches,warning_requests=sum(r.get('severity')!='critical' for r in requests),
        non_forward_requests=sum(r.get('source_id')!=report.get('forward_source_id','Q1') for r in requests),cleared_requests=sum(r.get('lifecycle_state')!='ACTIVE' for r in requests),
        missing_critical_decisions=len({e['event_id'] for r in ai for e in r.get('hazard_details',[]) if e['severity']=='critical'}-{r['event_id'] for r in timeline if r.get('stage')=='control_decision'}))
    event_validation=dict(transitions=sum(r.get('stage')=='event_transition' for r in timeline),
        critical_decisions=sum(r.get('stage')=='control_decision' for r in timeline),requests=len(requests),
        accepted=sum(r.get('stage')=='brake_accepted' for r in timeline),timeouts=sum(r.get('stage')=='brake_timeout' for r in timeline),
        unmatched_hazard=unmatched,confirm_to_ui_ms=stats([(first_ui[e]-t)/1e6 for e,t in confirms.items() if e in first_ui]),
        clear_to_ui_ms=stats([max(0,(clear_ui[e]-t)/1e6) for e,t in clears.items() if e in clear_ui]))
    severity=Counter(e['event_type']+':'+e['severity'] for r in ai for e in r.get('hazard_details',[]))
    result=dict(kind='ACTUAL_VIDEO_ADDITIONAL_EVIDENCE',timing_by_mode=details,window_rates=windows,early_late=decay,runtime_errors=errors,
        event_validation=event_validation,control_consistency=control_consistency,actual_event_result_counts=dict(severity),last_semantics=ai[-1].get('semantics'),last_rule_metrics=ai[-1].get('risk_rules'),
        baseline_method_rates=baseline_rates,source_counter_fps=source_fps,texture_counter_fps=texture_fps,
        baseline_fps_passed=(baseline_rates['ROAD_Q1']['completed_fps']>=6.5 and baseline_rates['MOSAIC_FULL']['completed_fps']>=13.8 and report['qt_scene_graph_fps_mean']>=28),
        qt_receive_apply_ms=stats([r.get('qt_total_apply_ms') for r in qt if lo<=r.get('qt_apply_ns',0)<=hi]),
        log_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ev.glob(label+'*') if p.suffix in ('.jsonl','.log')})
    out=ev/(label+'_additional.json');out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
