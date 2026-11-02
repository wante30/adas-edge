"""Run isolated real RKNN/Qt local replay; never changes installed services."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
BASE=Path('/dev/shm/adas_issue10_20261102')
RAW=Path('/home/cat/qt_gpu_probe_ai_20261019/local_quad_full.rgb')

def last_json(path):
    try:
        with path.open('rb') as f:
            f.seek(0,2);f.seek(max(0,f.tell()-65536));lines=f.read().splitlines()
        for line in reversed(lines):
            try:return json.loads(line)
            except ValueError:pass
    except FileNotFoundError:pass
    return {}

def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=8)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['baseline_src','candidate']);parser.add_argument('--label',required=True);parser.add_argument('--seconds',type=int,default=70);parser.add_argument('--model',default='FAST-Q4');parser.add_argument('--raw',default=str(RAW));parser.add_argument('--sequence',action='store_true');parser.add_argument('--no-bev-trace',action='store_true');args=parser.parse_args()
    root=BASE/args.kind;ev=BASE/'evidence';label=args.label
    env=dict(os.environ,DISPLAY=':0.0',XAUTHORITY='/home/cat/.Xauthority',QT_QPA_PLATFORM='xcb',QT_XCB_GL_INTEGRATION='xcb_egl',ADAS_EDGE_ROOT=str(root/'ai'),ADAS_DISABLE_ROAD_SCENE='1',ADAS_LEGACY_HAZARD='1',ADAS_AUTO_BRAKE='1',ADAS_AI_DEBUG='1',ADAS_CONTROL_MODE='DRY_RUN',ADAS_CONTROL_AUTHORIZED='0',ADAS_TEST_ACTION_PORT='38372',ADAS_AI_REQUEST_PORT='38270',ADAS_AI_RESULT_PORT='38271',ADAS_EVENT_TIMELINE=str(ev/(label+'_events.jsonl')))
    ai_args=['python3','-u',str(root/'tools/isolated_ai.py'),'--live-frame-file',str(BASE/'frames.mfr1'),'--source','LOCAL_REPLAY','--layout','quad','--display-view','front','--road-quadrant','3','--forward-quadrant','3','--navigation-quadrant','1','--main-model','/home/cat/models/b05n_p2_mtl_b2a_seg256/b05n_p2_mtl_b2a_seg256_int8.rknn','--main-hz','10','--hazard-hz','5','--ocr-hz','2','--navigation-ocr','--idle-timeout','0','--request-port','38270','--result-port','38271','--timing-log',str(ev/(label+'_ai.jsonl')),'--summary',str(ev/(label+'_ai_summary.json'))]
    # Test-only external QML selects a real model through the existing public Qt method.
    s=(root/'hmi/main.qml').read_text(encoding='utf-8');i=s.rfind('}');s=s[:i]+('\n Timer {interval:3500;running:true;repeat:false;onTriggered:video.requestModelSwitch("%s")}\n'%args.model)+s[i:]
    if args.sequence:
        i=s.rfind('}');s=s[:i]+"""
 Timer {interval:30000;running:true;repeat:false;onTriggered:root.viewMode=1}
 Timer {interval:40000;running:true;repeat:false;onTriggered:root.viewMode=0}
 Timer {interval:50000;running:true;repeat:false;onTriggered:video.requestModelSwitch("MTL-FAST")}
 Timer {interval:70000;running:true;repeat:false;onTriggered:video.requestModelSwitch("FAST-Q4")}
"""+s[i:]
    qml=root/'hmi'/(label+'_startup.qml');qml.write_text(s,encoding='utf-8')
    env.update(QT_QML_FILE=str(qml),QT_BEV_TRACE=str(ev/(label+'_trace.jsonl')),QT_BEV_MEASUREMENTS=str(ev/(label+'_measurements.jsonl')),QT_SCENE_TIMELINE_LOG=str(ev/(label+'_scene.jsonl')))
    if args.no_bev_trace:env.update(QT_BEV_TRACE='',QT_BEV_MEASUREMENTS='')
    hmi_args=[str(root/'build/qt_gpu_probe'),'--fps','60','--duration',str(args.seconds),'--start-mode','2','--pgv-layout','quad','--road-quadrant','3','--forward-quadrant','3','--raw-replay',args.raw,'--ai-frame-file',str(BASE/'frames.mfr1'),'--ai-hz','0','--metrics-jsonl',str(ev/(label+'_hmi.jsonl')),'--output-json',str(ev/(label+'_summary.json'))]
    processes=[];samples=[];start=time.monotonic();before=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
    try:
        for name,cmd,cwd in [('ai',ai_args,root/'ai'),('hmi',hmi_args,root/'build')]:
            with (ev/(label+'_'+name+'.log')).open('wb') as log:
                processes.append(subprocess.Popen(cmd,cwd=cwd,env=env,stdout=log,stderr=log,start_new_session=True))
        (ev/(label+'_processes.json')).write_text(json.dumps({'args':hmi_args,'ai_args':ai_args,'pids':[p.pid for p in processes],'source':'LOCAL_REPLAY','control_mode':'DRY_RUN','authorized':False,'board_clock':time.time(),'host_note':'board wall clock is not host date'},indent=2))
        while processes[1].poll() is None and (args.seconds==0 or time.monotonic()-start<args.seconds+20):
            time.sleep(1)
            after=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));total=sum(after)-sum(before);idle=after[3]+after[4]-before[3]-before[4]
            samples.append({'elapsed_s':time.monotonic()-start,'cpu_all_four_cores_percent':100*(1-idle/total) if total else 0});before=after
        if processes[1].poll() is None:raise RuntimeError('Qt timeout')
        if processes[1].returncode:raise RuntimeError('Qt failed: '+str(processes[1].returncode))
        print(json.dumps({'label':label,'elapsed':time.monotonic()-start,'last_metric':last_json(ev/(label+'_hmi.jsonl'))}),flush=True)
    finally:
        (ev/(label+'_cpu.json')).write_text(json.dumps(samples,indent=2))
        for proc in reversed(processes):stop(proc)

if __name__=='__main__':main()
