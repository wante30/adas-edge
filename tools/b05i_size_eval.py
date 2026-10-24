import json, sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO
root=Path(r"D:\Datasets\adas_edge_training\b05i_train_20261024\bdd50")
def iou(a,b):
 x1,y1,x2,y2=a; X1,Y1,X2,Y2=b; inter=max(0,min(x2,X2)-max(x1,X1))*max(0,min(y2,Y2)-max(y1,Y1)); aa=(x2-x1)*(y2-y1); bb=(X2-X1)*(Y2-Y1); return inter/(aa+bb-inter) if aa+bb-inter>0 else 0
def load_gt(p):
 rows=[]
 for line in p.read_text().splitlines():
  _,cx,cy,w,h=map(float,line.split()); rows.append([(cx-w/2)*512,(cy-h/2)*192,(cx+w/2)*512,(cy+h/2)*192])
 return rows
def main():
 models={'source':r'D:\Datasets\adas_edge_training\tl_specialist\tl-512x192\weights\best.pt','candidate':r'D:\Datasets\adas_edge_training\b05i_runs\candidate\weights\best.pt'}
 out={}
 for name,path in models.items():
  m=YOLO(path); res=m.predict(source=str(root/'images'), imgsz=512, rect=True, conf=.25, device=0, verbose=False, stream=True)
  stats={b:{'gt':0,'tp':0,'fp':0,'fn':0} for b in ('<8','8-16','16-32','>32')}
  for r,imgpath in zip(res,sorted((root/'images').glob('*.jpg'))):
   gt=load_gt(root/'labels'/(imgpath.stem+'.txt')); preds=[list(map(float,b.xyxy[0])) for b in r.boxes]
   used=set()
   for g in gt:
    w=g[2]-g[0]; b='<8' if w<8 else '8-16' if w<16 else '16-32' if w<32 else '>32'; stats[b]['gt']+=1
    best=0; bi=None
    for i,p in enumerate(preds):
     if i in used: continue
     o=iou(g,p)
     if o>best: best,bi=o,i
    if bi is not None and best>=.5: used.add(bi); stats[b]['tp']+=1
    else: stats[b]['fn']+=1
   stats['<8']['fp'] += 0
  # Count FP overall separately
  out[name]={b:{**v,'recall':v['tp']/v['gt'] if v['gt'] else None} for b,v in stats.items()}
 print(json.dumps(out,indent=2)); Path(r'C:\Users\JingyuWang\adas-edge\outputs\b05i_size_eval.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__': main()
