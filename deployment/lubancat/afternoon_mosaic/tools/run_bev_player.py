"""Record the actual board Qt renderer replaying one shared measurement file."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
BASE=Path('/dev/shm/adas_issue10_20261102')

def cpu():
    v=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));return sum(v),v[3]+v[4]

def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['baseline_src','candidate']);p.add_argument('--label',required=True);p.add_argument('--measurements',required=True);a=p.parse_args()
    ev=BASE/'evidence';root=BASE/a.kind
    env=dict(os.environ,DISPLAY=':0.0',XAUTHORITY='/home/cat/.Xauthority',QT_QPA_PLATFORM='xcb',QT_XCB_GL_INTEGRATION='xcb_egl',QT_BEV_TRACE=str(ev/(a.label+'_trace.jsonl')),QT_SCENE_TIMELINE_LOG=str(ev/(a.label+'_scene.jsonl')))
    cmd=[str(root/'build-replay/bev_replay'),a.measurements,'/home/cat/qt_gpu_probe_ai_20261019/local_quad_full.rgb',a.label]
    with (ev/(a.label+'_player.log')).open('wb') as log:
        proc=subprocess.Popen(cmd,cwd=root/'build-replay',env=env,stdout=log,stderr=log,start_new_session=True)
    with (ev/(a.label+'_record.log')).open('wb') as log:
        recorder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','x11grab','-framerate','12','-video_size','1280x720','-i',':0.0','-vf','scale=960:540','-c:v','libx264','-preset','ultrafast','-crf','20','-threads','1','-pix_fmt','yuv420p',str(ev/(a.label+'.mp4'))],env=env,stdin=subprocess.PIPE,stdout=log,stderr=log,start_new_session=True)
    samples=[];before=cpu();start=time.monotonic()
    try:
        while proc.poll() is None and time.monotonic()-start<100:
            time.sleep(1);after=cpu();total=after[0]-before[0];idle=after[1]-before[1]
            samples.append({'elapsed_s':time.monotonic()-start,'cpu_all_four_cores_percent':100*(1-idle/total) if total else 0});before=after
        if proc.poll() is None:raise RuntimeError('Player timeout')
        if proc.returncode:raise RuntimeError('Player failure '+str(proc.returncode))
    finally:
        if proc.poll() is None:proc.terminate();proc.wait()
        if recorder.poll() is None:
            recorder.communicate(b'q',timeout=10)
        (ev/(a.label+'_cpu.json')).write_text(json.dumps(samples,indent=2))
    print(a.label,'finished',round(time.monotonic()-start,2),'seconds',flush=True)
if __name__=='__main__':main()
