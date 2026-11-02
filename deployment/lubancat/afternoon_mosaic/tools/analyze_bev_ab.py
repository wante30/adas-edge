"""Analyze actual Qt renderer traces; coordinates are relative BEV units."""
import argparse,collections,csv,json,math
from pathlib import Path
import numpy as np

def read(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            try:yield json.loads(line)
            except ValueError:pass

def summarize(path):
    rows=list(read(path));groups=collections.defaultdict(list)
    for r in rows:groups[(r['source_key'],r['track_id'])].append(r)
    steps=[];speed=[];interval=[];lag=[];recreates=0;true_recreates=0;held=0;invalid=set();ids=set()
    for key,items in groups.items():
        extra=max(r['birth_count'] for r in items)-1
        recreates+=extra
        if items[0]['server_track_id']>0:true_recreates+=extra
        for r in items:
            if r['server_track_id']<=0:invalid.add(key)
            else:ids.add((key[0],r['server_track_id']))
            held+=r['measurement_age_s']>.6
            lag.append(math.hypot(r['raw_x']-r['filtered_x'],r['raw_z']-r['filtered_z']))
        for a,b in zip(items,items[1:]):
            dt=(b['render_ns']-a['render_ns'])/1e9
            if dt<=0 or dt>.15 or b['birth_count']!=a['birth_count']:continue
            if b['server_track_id']<=0:continue
            delta=math.hypot(b['filtered_x']-a['filtered_x'],b['filtered_z']-a['filtered_z'])
            steps.append(delta/ (60*dt));speed.append(delta/dt);interval.append(dt)
    def stats(xs):
        return {k:float(v) for k,v in zip(('p50','p95','p99','max'),np.percentile(xs,[50,95,99,100]))} if xs else {}
    return {'trace_rows':len(rows),'render_seconds':(max(r['render_ns'] for r in rows)-min(r['render_ns'] for r in rows))/1e9 if rows else 0,'true_identity_count':len(ids),'untracked_rendered_identities':len(invalid),'recreate_count':recreates,'true_identity_recreate_count':true_recreates,'held_after_600ms_rows':held,'step_at_60hz':stats(steps),'speed_relative_units_per_s':stats(speed),'visual_frame_interval_s':stats(interval),'display_to_raw_distance':stats(lag),'steps_over_05_at_60hz':sum(x>.5 for x in steps),'steps_over_1_at_60hz':sum(x>1 for x in steps),'valid_step_count':len(steps)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('traces',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
    output={Path(path).stem:summarize(path) for path in a.traces}
    Path(a.output).write_text(json.dumps(output,indent=2),encoding='utf-8');print(json.dumps(output,indent=2))
