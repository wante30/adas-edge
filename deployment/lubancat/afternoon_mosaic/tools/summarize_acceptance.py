"""Summarize a real acceptance run, keeping full and trimmed steady windows distinct."""
import argparse,json
from pathlib import Path
from analyze_performance import analyze_rows,dist


def summarize(system,acceptance):
    lines=[json.loads(l) for l in Path(system).read_text(encoding='utf-8').splitlines()]
    report=json.loads(Path(acceptance).read_text(encoding='utf-8'))
    samples=[r for r in lines if r.get('kind')=='sample']
    result={'full':analyze_rows(lines,str(system)), 'steady_by_model':{}, 'acceptance':report,
            'method':'Full includes switches. Steady excludes first 10s and last 2s of each model window. AI FPS uses actual completion intervals per bridge/epoch; CPU is four-core system busy.'}
    for model in ('MTL-FAST','FAST-Q4','FAST-MOSAIC'):
        windows=[(w['start_ns']+10_000_000_000,w['end_ns']-2_000_000_000)
                 for w in report['windows'] if w['model']==model]
        selected=[r for r in samples if any(a<=r['timestamp_ns']-int(r['dt_s']*1e9) and r['timestamp_ns']<=b for a,b in windows)
                  and all(h['model_id']==model and h['inference_ready'] for h in r['hmi'])]
        if selected:result['steady_by_model'][model]=analyze_rows([lines[0]]+selected,model,contiguous=False)
    result['frequencies']={k:dist(r.get('frequencies',{}).get(k) for r in samples)
                           for k in ('cpu_khz','gpu_hz','npu_hz','dram_hz','temperature_millic')}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('system');p.add_argument('acceptance');p.add_argument('--output',required=True);a=p.parse_args()
    result=summarize(a.system,a.acceptance)
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,r in [('FULL',result['full'])]+list(result['steady_by_model'].items()):
        print(name,'seconds',round(r['seconds'],2),'Qt',r['hmi']['scene_graph_fps'],'AI',dict(r['ai']))
