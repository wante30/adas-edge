"""Run via SSH stdin. Screenshots stream to the PC; no batch images on board."""
import base64,ctypes as C,hashlib,json,os,pathlib,signal,struct,subprocess,sys,tempfile,time,zlib
ROOT=pathlib.Path('/home/cat/adas_issue9_20261102/src')
EV=ROOT/'evidence'
STATE=json.loads((EV/'integration_session.json').read_text())
LABEL=STATE['label']
ENV=dict(os.environ,DISPLAY=':0',XAUTHORITY='/home/cat/.Xauthority')
AI=EV/(LABEL+'_ai.jsonl');HMI=EV/(LABEL+'_hmi.jsonl')
stop=False

def emit(kind,**data):
    print(json.dumps(dict(kind=kind,**data),separators=(',',':')),flush=True)

def last(path):
    try:
        with pathlib.Path(path).open('rb') as f:
            f.seek(0,2);f.seek(max(0,f.tell()-131072));lines=f.read().splitlines()
        for line in reversed(lines):
            try:return json.loads(line)
            except ValueError:pass
    except (OSError,ValueError):pass
    return {}

def compact_ai():
    a=last(AI)
    return {k:a.get(k) for k in ('frame_id','model_id','ai_fps','capture_timestamp_ns','result_publish_timestamp_ns','inference','detection_counts','hazard_details','event_updates','control','workload','traffic','navigation_state')}

def vehicle():return last('/dev/shm/adas_vehicle_status.json')

def click(x,y):
    subprocess.run(['/usr/bin/python3',str(ROOT/'tools/ui_control.py'),str(x),str(y)],env=ENV,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)

def close_overlay():click(1222,78);time.sleep(.12)

def select(model):
    # Open the actual visible model picker, then click the matching row.
    click(1200,40);time.sleep(.18)
    click(1000,{'MTL-FAST':304,'FAST-Q4':348,'FAST-MOSAIC':392,'FAST-MOSAIC-LITE':436}[model])
    end=time.monotonic()+55
    while time.monotonic()<end:
        a=last(AI);h=last(HMI)
        if a.get('model_id')==model and h.get('model_id')==model and h.get('ai_online') and h.get('inference_ready'):
            time.sleep(1.0);return
        time.sleep(.25)
    raise RuntimeError('Model switch not confirmed: '+model)

counter=0
TMP=None

def capture(section,overlay='none',probe=None):
    global counter
    counter+=1
    before=time.monotonic_ns();a=compact_ai();v=vehicle();h=last(HMI)
    file=pathlib.Path(TMP)/'capture.png'
    r=subprocess.run(['xfce4-screenshooter','--fullscreen','--save',str(file)],env=ENV,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15)
    if r.returncode or not file.exists():raise RuntimeError(r.stderr.decode(errors='replace'))
    png=file.read_bytes();file.unlink();after=time.monotonic_ns()
    meta=dict(sequence=counter,section=section,overlay=overlay,source_type='REAL_PCIE_LIVE',
        boot_id=STATE['boot_id'],session=LABEL,start_monotonic_ns=before,end_monotonic_ns=after,
        ai_before=a,ai_after=compact_ai(),vehicle_before=v,vehicle_after=vehicle(),
        hmi={k:h.get(k) for k in ('model_id','source_mode','ai_online','inference_ready','ai_frame_id','ai_fps','fps','source_fps','produced_frames')},
        probe_label=probe,sha256=hashlib.sha256(png).hexdigest())
    emit('frame',metadata=meta,png=base64.b64encode(png).decode('ascii'))


def series(section,count,interval=1.2,overlay='none'):
    emit('stage',section=section,count=count)
    for i in range(count):
        capture(section,overlay=overlay)
        time.sleep(interval)


def quadrants():
    import numpy as np
    samples=[];previous=None
    for i in range(4):
        b=pathlib.Path('/dev/shm/pango-video/latest.pgv').read_bytes()
        magic,version,w,h,fid,gen,crc,length,lo,hi=struct.unpack('<4s9I',b[:40]);raw=b[40:]
        valid=magic==b'PGVF' and version==1 and len(raw)==length and zlib.crc32(raw)&0xffffffff==crc
        p=np.frombuffer(raw,dtype='<u2').reshape(h,w);q=[]
        for n in range(4):
            y,x=(n//2)*360,(n%2)*640;a=p[y:y+360,x:x+640]
            q.append(dict(source_id='Q%d'%(n+1),nonzero_fraction=float(np.count_nonzero(a)/a.size),changed_pixels=int(np.count_nonzero(a!=previous[y:y+360,x:x+640])) if previous is not None else None))
        samples.append(dict(frame_id=fid,generation=gen,crc_valid=valid,quadrants=q));previous=p.copy();time.sleep(.25)
    return samples


def provenance():
    paths=['hmi/main.qml','build-stage4/qt_gpu_probe','ai/ai_bridge.py','ai/auto_brake.py','ai/light_risk.py','hardware/hardware_bridge.py']
    return dict(boot_id=STATE['boot_id'],state=STATE,files=[dict(path=str(ROOT/p),sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in paths],quadrant_samples=quadrants(),capture_tool='xfce4-screenshooter --fullscreen --save',screenshot_size=[1280,720],board_clock_untrusted=True)


def probe():
    label='assets_control_'+str(time.monotonic_ns())
    emit('stage',section='controlled_fpga_probe',count=6)
    with tempfile.TemporaryFile() as log:
        p=subprocess.Popen(['/usr/bin/python3',str(ROOT/'tools/validate_real_control.py'),'--label',label,'--count','1'],stdout=log,stderr=log)
        end=time.monotonic()+1.6
        while time.monotonic()<end:
            v=vehicle()
            if v.get('current_action')=='brake' and v.get('brake_allow'):break
            time.sleep(.02)
        for i in range(6):capture('fpga_control',probe=label)
        p.wait(timeout=10)
        report=json.loads((EV/(label+'_manual_control.json')).read_text())
        emit('hardware_probe_report',report=report,label=label)
        if not report.get('passed'):emit('warning',message='Controlled probe failed; exclude probe screenshots from curated and core')


def main():
    global TMP
    emit('provenance',report=provenance())
    original=last(AI).get('model_id','MTL-FAST')
    with tempfile.TemporaryDirectory(prefix='adas-assets-',dir='/dev/shm') as tmp:
        TMP=tmp
        try:
            select('MTL-FAST')
            probe()
            series('mtl_front',100)
            select('FAST-Q4');series('fast_q4',50)
            select('FAST-MOSAIC');series('fast_mosaic',65)
            select('FAST-MOSAIC-LITE');series('mosaic_lite',15)
            select('MTL-FAST');click(574,32);time.sleep(2);series('tesla_bev',35)
            click(392,32);time.sleep(.5)
            for model in ('MTL-FAST','FAST-Q4','FAST-MOSAIC','FAST-MOSAIC-LITE'):
                select(model);click(1200,40);time.sleep(.4)
                series('model_menu',4,interval=.7,overlay='model_menu')
                click(1222,78);time.sleep(.2)
            select('MTL-FAST');click(790,88);time.sleep(.5);click(120,538);time.sleep(.7)
            series('fpga_register_read',6,interval=.8,overlay='fpga_read_panel')
            click(900,585)
        finally:
            try:
                # Escape closes overlays via the app's real keyboard handler.
                x=C.CDLL('libX11.so.6');t=C.CDLL('libXtst.so.6')
                x.XOpenDisplay.argtypes=[C.c_char_p];x.XOpenDisplay.restype=C.c_void_p
                x.XKeysymToKeycode.argtypes=[C.c_void_p,C.c_ulong];x.XKeysymToKeycode.restype=C.c_uint
                x.XFlush.argtypes=[C.c_void_p];x.XCloseDisplay.argtypes=[C.c_void_p]
                t.XTestFakeKeyEvent.argtypes=[C.c_void_p,C.c_uint,C.c_int,C.c_ulong]
                d=x.XOpenDisplay(b':0');key=x.XKeysymToKeycode(d,0xff1b)
                t.XTestFakeKeyEvent(d,key,1,0);t.XTestFakeKeyEvent(d,key,0,0);x.XFlush(d);x.XCloseDisplay(d)
                time.sleep(.15);select(original if original in ('MTL-FAST','FAST-Q4','FAST-MOSAIC','FAST-MOSAIC-LITE') else 'MTL-FAST')
                emit('restored',model=last(AI).get('model_id'),control=last(AI).get('control'))
            except Exception as exc:emit('restore_error',message=str(exc))
    emit('complete',frames=counter)

try:main()
except Exception as exc:
    emit('error',message=repr(exc));raise
