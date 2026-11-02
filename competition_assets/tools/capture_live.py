"""PC side of the stream collector; resumes file numbering without overwriting."""
import base64,datetime,hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
RAW=ROOT/'raw_candidates'/'live_pcie';RAW.mkdir(parents=True,exist_ok=True)
script=pathlib.Path(__file__).with_name('remote_collector.py').read_bytes()
log=(ROOT/'evidence'/'remote_stderr.log').open('ab')
p=subprocess.Popen(['ssh','-4','-o','BatchMode=yes','-o','ConnectTimeout=10','cat@192.168.137.2','python3 -'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log)
p.stdin.write(script);p.stdin.close()
with (ROOT/'evidence'/'acquisition.jsonl').open('a',encoding='utf-8',buffering=1) as records:
    for line in iter(p.stdout.readline,b''):
        row=json.loads(line);row['pc_received_at']=datetime.datetime.now().astimezone().isoformat()
        if row['kind']=='frame':
            png=base64.b64decode(row.pop('png'));meta=row['metadata']
            assert hashlib.sha256(png).hexdigest()==meta['sha256']
            file=RAW/('%s_%04d.png'%(meta['section'],meta['sequence']))
            with file.open('xb') as out:out.write(png)
            row['file']=str(file.relative_to(ROOT)).replace('\\','/')
            if meta['sequence']%20==0:print('saved',meta['sequence'],meta['section'],flush=True)
        else:
            print(row['kind'],json.dumps({k:v for k,v in row.items() if k not in ('report',)},ensure_ascii=False)[:600],flush=True)
            if row['kind']=='provenance':(ROOT/'evidence'/'live_runtime_provenance.json').write_text(json.dumps(row['report'],indent=2),encoding='utf-8')
            if row['kind']=='hardware_probe_report':(ROOT/'evidence'/'controlled_fpga_probe.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        records.write(json.dumps(row,ensure_ascii=False)+'\n')
p.wait();log.close();print('collector_exit',p.returncode,flush=True)
sys.exit(p.returncode)
