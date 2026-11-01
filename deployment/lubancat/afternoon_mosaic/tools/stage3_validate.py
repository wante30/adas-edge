"""Capture actual staged validation without changing production services."""
import json, subprocess, time
from pathlib import Path
ROOT=Path('/home/cat/adas_mosaic_dev_20261101')
def click(x,y):
    subprocess.run(['python3',str(ROOT/'tools/ui_control.py'),str(x),str(y)],check=True)
def capture(name):
    subprocess.run(['xwd','-display',':0','-root','-silent','-out',str(ROOT/'evidence'/f'{name}.xwd')],env=dict(__import__('os').environ, XAUTHORITY='/home/cat/.Xauthority'),check=True)
def recent(label):
    lines=(ROOT/'evidence'/f'{label}_ai.jsonl').read_text().splitlines()
    rows=[json.loads(x) for x in lines[-250:]]
    last=rows[-1]; mode=last['inference']['mode']; epoch=last['inference']['epoch']
    rows=[x for x in rows if x['inference']['epoch']==epoch]
    span=(rows[-1]['result_publish_timestamp_ns']-rows[0]['result_publish_timestamp_ns'])/1e9
    return dict(mode=mode,epoch=epoch,results=len(rows),fps=(len(rows)-1)/span if span else 0,boxes=last['boxes'],navigation=last['navigation_state'],main_timing=last['main_timing'])
steps=[]
for x in [390,480,390,480,390]:
    click(x,32);time.sleep(6);steps.append(recent('stage3')); print(json.dumps(steps[-1]),flush=True)
(ROOT/'evidence/stage3_switches.json').write_text(json.dumps(steps,indent=2))
capture('stage3_road_final')
