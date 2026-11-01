"""Exercise the visible HMI, then retain 10 minutes of continuous replay evidence."""
import json, os, subprocess, time, traceback
from pathlib import Path
from test_session import ROOT, EVIDENCE, STATE, record, stop, launch, save
LABEL=os.environ.get('ADAS_VALIDATION_LABEL','final_replay')
report={'input':'local-raw 1280x720 RGB565, requested 60 FPS','source_path':'/home/cat/qt_gpu_probe_ai_20261019/local_quad_full.rgb','events':[]}
def last_rows(path,size=200000):
    with path.open('rb') as f:
        f.seek(0,2); f.seek(max(0,f.tell()-size)); data=f.read().splitlines()
    out=[]
    for line in data:
        try:out.append(json.loads(line))
        except ValueError:pass
    return out
def metric():return last_rows(EVIDENCE/(LABEL+'_hmi.jsonl'),16000)[-1]
def click(x,y):
    subprocess.run(['python3',str(ROOT/'tools/ui_control.py'),str(x),str(y)],check=True)
def wait_for(predicate,timeout=30):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        try:
            value=metric()
            if predicate(value):return value
        except (FileNotFoundError,IndexError):pass
        time.sleep(.2)
    raise RuntimeError('HMI condition timed out: '+json.dumps(metric()))
def capture(name):
    subprocess.run(['xwd','-display',':0','-root','-silent','-out',str(EVIDENCE/(name+'.xwd'))],env=dict(os.environ,XAUTHORITY='/home/cat/.Xauthority'),check=True)
def event(name):
    value=metric(); value['event']=name;value['monotonic_ns']=time.monotonic_ns()
    rows=last_rows(EVIDENCE/(LABEL+'_ai.jsonl'))
    if rows:
        epoch=rows[-1]['inference']['epoch']; model=rows[-1].get('model_id')
        rows=[r for r in rows if r['inference']['epoch']==epoch and r.get('model_id')==model]
        if len(rows)>1:
            span=(rows[-1]['result_publish_timestamp_ns']-rows[0]['result_publish_timestamp_ns'])/1e9
            value['completed_fps']=(len(rows)-1)/span if span else 0
        value['last_ai']=rows[-1]
    vehicle=Path('/dev/shm/adas_vehicle_status.json')
    value['vehicle']=json.loads(vehicle.read_text()) if vehicle.exists() else {'online':False,'reason':'PCIe endpoint absent'}
    report['events'].append(value)
    (EVIDENCE/'stage5_acceptance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in value.items() if k not in ('last_ai','vehicle')}),flush=True)
try:
    wait_for(lambda m:m['ai_online'] and m['ai_results']>20,90)
    for model,y in [('B',172),('B-capi',216),('MTL-FULL',260),('MTL-FAST',304)]:
        before=metric()['ai_results'];click(600,710);click(1200,40);time.sleep(.4);click(1000,y)
        wait_for(lambda m:m['model_id']==model and m['ai_online'] and m['ai_results']>before+20,60)
        event('model_'+model);capture('stage5_model_'+model)
    for index,x in enumerate([390,480,390,480,390]):
        before=metric()['ai_results'];click(x,32)
        mode='ROAD_Q1' if x==390 else 'MOSAIC_FULL'
        wait_for(lambda m:m['inference_mode']==mode and m['inference_ready'] and m['ai_online'] and m['ai_results']>before+30)
        event('mode_'+str(index)+'_'+mode);capture('stage5_mode_'+str(index))
    click(570,32);time.sleep(2);capture('stage5_bev');click(390,32)
    # Stop only the owned AI process; the same HMI must show 0 FPS and reconnect.
    state=json.loads(STATE.read_text()); old=record(state['test_ai_pid']);before=metric()['ai_results']
    stop(state['test_ai_pid'])
    wait_for(lambda m:not m['ai_online'] and m['ai_fps']==0,10);event('ai_offline_zero_fps');capture('stage5_offline')
    state['test_ai_pid']=launch(old['args'],old['cwd'],old['env'],LABEL+'_ai.log');save(state)
    wait_for(lambda m:m['ai_online'] and m['ai_results']>before+20,40);event('ai_reconnected')
    # Continuous run begins after intentional failure recovery.
    start=time.monotonic();report['stability_start_ns']=time.monotonic_ns()
    hmi_pid=state['test_hmi_pid'];ai_pid=state['test_ai_pid'];mode_x=390
    for n in range(61):
        assert Path('/proc/'+str(hmi_pid)).exists() and Path('/proc/'+str(ai_pid)).exists()
        if n and n%6==0:
            mode_x=480 if mode_x==390 else 390;click(mode_x,32)
            wait_for(lambda m:m['inference_ready'] and m['ai_online'])
        event('stability_'+str(n))
        if n in (0,30,60):capture('stage5_stability_'+str(n))
        if n<60:time.sleep(10)
    report['continuous_seconds']=time.monotonic()-start
    report['stability_end_ns']=time.monotonic_ns()
    report['passed']=report['continuous_seconds']>=600
    report['window']=subprocess.check_output(['xwininfo','-display',':0','-name','Qt Quick GPU Probe - RK3566'],env=dict(os.environ,XAUTHORITY='/home/cat/.Xauthority')).decode()
    assert 'IsViewable' in report['window']
    event('finished');print('ACCEPTANCE_COMPLETE',flush=True)
except BaseException:
    report['passed']=False;report['error']=traceback.format_exc()
    (EVIDENCE/'stage5_acceptance.json').write_text(json.dumps(report,indent=2));print(report['error'],flush=True);raise
