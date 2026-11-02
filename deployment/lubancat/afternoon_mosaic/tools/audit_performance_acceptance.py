"""Independent post-run checks over recorded evidence, including actual broker stage names."""
import argparse,json
from collections import Counter
from pathlib import Path
from validate_followup import rows

def audit(directory,label):
    root=Path(directory);report=json.loads((root/(label+'_performance_acceptance.json')).read_text())
    start,end=report['started_ns'],report['ended_ns']
    events=[r for r in rows(root/(label+'_events.jsonl')) if start<=r.get('logged_ns',0)<=end]
    ai=[r for r in rows(root/(label+'_ai.jsonl')) if start<=r.get('result_publish_timestamp_ns',0)<=end]
    ui=[r for r in rows(root/(label+'_risk_ui.jsonl')) if start<=r.get('ui_applied_ns',0)<=end]
    system=[r for r in rows(root/(label+'_system.jsonl')) if r.get('kind')=='sample']
    sent=[e for e in events if e.get('stage') in ('brake_requested','control_request')]
    decisions=[e for e in events if e.get('stage')=='control_decision']
    checks=dict(elapsed_seconds=(end-start)/1e9,ai_rows=len(ai),ui_rows=len(ui),samples=len(system),
        models=dict(Counter(r['model_id'] for r in ai)),event_stages=dict(Counter(e.get('stage') for e in events)),
        decision_reasons=dict(Counter(e.get('reason') for e in decisions)),real_control_requests=len(sent),
        event_kinds=dict(Counter(e.get('kind') for e in events if e.get('stage')=='event_transition')),
        capture_offline_samples=sum(r.get('capture',{}).get('state')!='running' for r in system),
        lvt1_offline_samples=sum(not r.get('lvt1',{}).get('online') for r in system),
        wrong_forward_rows=sum(r.get('forward_source_id')!='Q4' for r in ai),
        stale_mtl_fast_q4=sum(bool((r.get('semantics') or {}).get('mtl_available')) for r in ai if r['model_id']=='FAST-Q4'),
        marked_boxes=sum(len(r.get('marked_boxes',[])) for r in ui),
        unmatched_hazard_samples=sum(bool(r.get('unmatched_hazard')) for r in ui))
    mismatch=0
    for row in ui:
        eventmap={e['event_id']:e for e in row.get('events',[])}
        for box in row.get('marked_boxes',[]):
            event=eventmap.get(box['event_id'])
            mismatch+=not event or event['severity']!=box['severity']
    checks['box_event_mismatches']=mismatch
    checks['passed']=bool(ai and ui and system) and checks['elapsed_seconds']>=600 and set(checks['models'])=={'MTL-FAST','FAST-Q4','FAST-MOSAIC'} and not any(checks[k] for k in ('real_control_requests','capture_offline_samples','lvt1_offline_samples','wrong_forward_rows','stale_mtl_fast_q4','box_event_mismatches'))
    return checks

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('label');p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(a.directory,a.label);Path(a.output).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)
