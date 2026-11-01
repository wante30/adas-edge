"""Derive final rates from monotonic completion/counter records, never UI guesses."""
import argparse, hashlib, json, statistics
from collections import Counter, defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--label',default='final_replay');p.add_argument('--output',type=Path);a=p.parse_args()
def read(name):return [json.loads(x) for x in (a.directory/name).read_text(encoding='utf-8').splitlines() if x.strip()]
r=json.loads((a.directory/'stage5_acceptance.json').read_text(encoding='utf-8'))
assert r.get('passed') and r['continuous_seconds']>=600, 'Continuous acceptance not complete'
rows=read(a.label+'_ai.jsonl');metrics=read(a.label+'_hmi.jsonl')
events=[e for e in r['events'] if e['event'].startswith('stability_')]
first,last=events[0],events[-1];seconds=(last['process_elapsed_ms']-first['process_elapsed_ms'])/1000
stable=[x for x in rows if r['stability_start_ns']<=x['result_publish_timestamp_ns']<=r['stability_end_ns']]
qt=[x for x in metrics if first['process_elapsed_ms']<=x['process_elapsed_ms']<=last['process_elapsed_ms']]
assert all(b['ai_results']>x['ai_results'] and b['source_frame_id']>x['source_frame_id'] for x,b in zip(events,events[1:])), 'Stopped data in continuous samples'
keys=[(x['inference']['bridge_id'],x['inference']['epoch'],x['frame_id']) for x in rows]
assert len(keys)==len(set(keys)), 'Duplicate published results'
def rates(data):
    runs=[]
    for x in data:
        key=(x['inference']['bridge_id'],x['inference']['epoch'],x.get('model_id'))
        if not runs or runs[-1][0]!=key:runs.append((key,[]))
        runs[-1][1].append(x)
    grouped=defaultdict(list)
    for key,items in runs:
        if len(items)>1:grouped[(items[-1]['inference']['mode'],key[-1])].append(items)
    out={}
    for (mode,model),groups in grouped.items():
        n=sum(len(g)-1 for g in groups);span=sum((g[-1]['result_publish_timestamp_ns']-g[0]['result_publish_timestamp_ns'])/1e9 for g in groups)
        flattened=[x for g in groups for x in g]
        out[mode+'/'+str(model)]={'completed_fps':n/span,'results':sum(map(len,groups)),
            'mean_pipeline_ms':statistics.mean(x['pipeline_ms'] for x in flattened),
            'mean_rknn_ms':statistics.mean(x['main_timing']['rknn_call_ms'] for x in flattened if x.get('main_timing'))}
    return out
log=(a.directory/(a.label+'_ai.log')).read_text(encoding='utf-8')
hmi=(a.directory/(a.label+'_hmi.log')).read_text(encoding='utf-8')
errors=[x for x in (log+'\n'+hmi).splitlines() if any(s in x for s in ('INFER_ERROR','REQUEST_LISTENER_ERROR','double free','corrupted','Segmentation fault','Aborted','Traceback'))]
assert not errors, errors
offline=next(e for e in r['events'] if e['event']=='ai_offline_zero_fps')
reconnected=next(e for e in r['events'] if e['event']=='ai_reconnected')
assert not offline['ai_online'] and offline['ai_fps']==0, 'AI offline state was not observed'
assert reconnected['last_ai']['inference']['bridge_id']!=offline['last_ai']['inference']['bridge_id'], 'AI instance did not change'
continuity=all(x['process_elapsed_ms']<=b['process_elapsed_ms'] and x['ai_results']<=b['ai_results'] for x,b in zip(r['events'],r['events'][1:]))
assert continuity, 'HMI counters reset across acceptance events'
summary={'input':r['input'],'source_path':r['source_path'],'continuous_seconds':r['continuous_seconds'],
 'window_is_viewable':'IsViewable' in r['window'],'hmi_counter_continuity_through_ai_restart':continuity,
 'counter_span_seconds':seconds,'source_fps':(last['produced_frames']-first['produced_frames'])/seconds,
 'texture_upload_fps':(last['uploaded_frames']-first['uploaded_frames'])/seconds,
 'qt_scene_graph_fps_mean':statistics.mean(x['scene_graph_fps'] for x in qt),
 'accepted_ai_fps_across_modes':(last['ai_results']-first['ai_results'])/seconds,
 'cpu_percent_all_cores_mean':statistics.mean(x['cpu_percent_of_all_cores'] for x in qt),
 'gpu_load_percent_mean':statistics.mean(x['gpu_load_percent'] for x in qt),
 'npu_driver_reported_percent_mean':statistics.mean(x['npu_load_percent'] for x in qt),
 'steady_mode_rates':rates(stable),'all_model_rates':rates(rows),
 'navigation_states':dict(Counter(x['navigation_state'] for x in stable)),
 'hazard_result_frames':sum(bool(x['hazards']) for x in stable),
 'normal_brake_attempts':sum(x.startswith(('AUTO_BRAKE ','AUTO_BRAKE_FAILED ')) for x in log.splitlines()),
 'normal_brake_timeouts':sum(x.startswith('AUTO_BRAKE_FAILED timeout') for x in log.splitlines()),
 'live_pcie_available':False,'lvt1_available':False,
 'runtime_errors':errors,'duplicate_result_keys':len(keys)-len(set(keys)),
 'mode_events':[{'event':e['event'],'mode':e['inference_mode'],'model':e['model_id'],'fps':e['ai_fps']} for e in r['events'] if not e['event'].startswith('stability_')],
 'log_sha256':{name:hashlib.sha256((a.directory/name).read_bytes()).hexdigest() for name in (a.label+'_ai.jsonl',a.label+'_hmi.jsonl',a.label+'_ai.log',a.label+'_hmi.log')}}
text=json.dumps(summary,ensure_ascii=False,indent=2)
if a.output:a.output.write_text(text,encoding='utf-8')
print(text)
