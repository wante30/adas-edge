"""Lossless X11 bursts around naturally occurring measured critical events."""
import base64,datetime,hashlib,json,pathlib,subprocess,struct,sys
import numpy as np
from PIL import Image
ROOT=pathlib.Path(__file__).resolve().parents[1]
remote=(pathlib.Path(__file__).with_name('remote_collector.py').read_text(encoding='utf-8').split('try:main()')[0])
remote += r'''
select('FAST-Q4')
try:
    seen=set();count=293;bursts=0;deadline=time.monotonic()+135
    while time.monotonic()<deadline and bursts<6:
        a=compact_ai();critical=[e for e in a.get('hazard_details') or [] if isinstance(e,dict) and e.get('severity')=='critical' and e.get('valid') and e.get('lifecycle_state')=='ACTIVE']
        fresh=[e for e in critical if e.get('event_id') not in seen]
        if not fresh:time.sleep(.035);continue
        for e in fresh:seen.add(e['event_id'])
        bursts+=1
        for j in range(4):
            count+=1;before=time.monotonic_ns();aa=compact_ai();vv=vehicle()
            xwd=subprocess.run(['xwd','-display',':0','-root','-silent'],env=ENV,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=5).stdout
            after=time.monotonic_ns()
            meta=dict(sequence=count,section='natural_hazard',overlay='none',source_type='REAL_PCIE_LIVE',
                boot_id=STATE['boot_id'],session=LABEL,start_monotonic_ns=before,end_monotonic_ns=after,
                ai_before=aa,ai_after=compact_ai(),vehicle_before=vv,vehicle_after=vehicle(),
                natural_critical_trigger_ids=[e['event_id'] for e in fresh],capture_method='xwd root; lossless RGB decoding on PC',
                xwd_sha256=hashlib.sha256(xwd).hexdigest(),burst=bursts,burst_frame=j)
            emit('xwd_frame',metadata=meta,xwd=base64.b64encode(xwd).decode('ascii'));time.sleep(.06)
    emit('burst_complete',bursts=bursts,frames=count-293)
finally:
    select('MTL-FAST');emit('restored',model=last(AI).get('model_id'),control=last(AI).get('control'))
'''
log=(ROOT/'evidence/burst_remote_stderr.log').open('ab')
p=subprocess.Popen(['ssh','-4','-o','BatchMode=yes','-o','ConnectTimeout=10','cat@192.168.137.2','python3 -'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log)
p.stdin.write(remote.encode());p.stdin.close()
folder=ROOT/'evidence/xwd_sources';folder.mkdir(exist_ok=True)
with (ROOT/'evidence/acquisition.jsonl').open('a',encoding='utf-8',buffering=1) as records:
    for line in iter(p.stdout.readline,b''):
        r=json.loads(line);r['pc_received_at']=datetime.datetime.now().astimezone().isoformat()
        if r['kind']=='xwd_frame':
            xwd=base64.b64decode(r.pop('xwd'));m=r['metadata'];header=struct.unpack('>25I',xwd[:100]);hs,ver,fmt,depth,w,h,xoff,order,unit,bitorder,pad,bpp,stride,vc,rm,gm,bm,bits,entries,colors,*rest=header
            assert ver==7 and fmt==2 and (w,h)==(1280,720) and bpp==32 and xoff==0
            offset=hs+colors*12;assert len(xwd)>=offset+h*stride
            pixels=np.frombuffer(xwd[offset:offset+h*stride],dtype='u1').reshape(h,stride)[:,:w*4].copy().view('<u4' if order==0 else '>u4').reshape(h,w)
            channels=[]
            for mask in (rm,gm,bm):
                shift=(mask&-mask).bit_length()-1;maximum=mask>>shift
                channels.append((((pixels&mask)>>shift)*255//maximum).astype('uint8'))
            im=Image.fromarray(np.stack(channels,axis=-1),'RGB');file=ROOT/'raw_candidates/live_pcie'/('natural_hazard_%04d.png'%m['sequence']);assert not file.exists();im.save(file,compress_level=6)
            source=folder/('natural_hazard_%04d.xwd'%m['sequence']);source.write_bytes(xwd)
            m['sha256']=hashlib.sha256(file.read_bytes()).hexdigest();m['native_xwd_source']=source.relative_to(ROOT).as_posix()
            r.update(kind='frame',file=file.relative_to(ROOT).as_posix());print('natural_hazard',m['sequence'],'capture_ms',round((m['end_monotonic_ns']-m['start_monotonic_ns'])/1e6),flush=True)
        else:print(r,flush=True)
        records.write(json.dumps(r,ensure_ascii=False)+'\n')
p.wait();log.close();raise SystemExit(p.returncode)