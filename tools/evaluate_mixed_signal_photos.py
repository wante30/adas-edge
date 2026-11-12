"""Fresh detector and manual-ROI evaluation on genuine mixed-light photos."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.evaluate_signals_offline import crop,dump,score_attr,text
from tools.offline_signal_models import OfflineDetector,sha256
from tools.report_signal_diagnostics import manual_result,observed_score
from competition_ai.signal_shape import ContourTemplateBackend
from competition_ai.multi_signal import MultiSignalTracker,Candidate
from competition_ai.signal_association import associate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    labels=json.loads((ROOT/'docs/multi_signal/offline_mixed_photos.json').read_text(encoding='utf-8'))
    existing=json.loads((ROOT/'docs/multi_signal/photo_labels.json').read_text(encoding='utf-8'))
    rows=labels['images']+[dict(row,path='outputs/real_signal_photos/'+row['id']+'.jpg',split='PREVIOUSLY_REVIEWED') for row in existing['images']]
    paths=dict(bdd='rknn_b05o/bdd_yolov8n_512x288_raw_heads.onnx',p2='rknn_b05o/b05n_p2_multiclass_512x288_raw_heads_20261026.onnx')
    models={k:OfflineDetector(Path('D:/Datasets/adas_edge_training')/v,traffic_light_id=8) for k,v in paths.items()}
    dump(args.output/'models.json',{k:m.evidence for k,m in models.items()})
    backend=ContourTemplateBackend();counts={};alltiles=[];failures=args.output/'failures';failures.mkdir()
    with (args.output/'frames.jsonl').open('x',encoding='utf-8') as stream:
        for fid,row in enumerate(rows):
            path=ROOT/row['path']
            if sha256(path)!=row['sha256']:raise ValueError('Photo hash mismatch')
            image=cv2.imread(str(path));group=counts.setdefault(row['split'],Counter());out=dict(image_id=row['id'],split=row['split'],source_sha256=row['sha256'],manual={},detected={})
            for res,im in [('original',image),('q4',cv2.resize(image,(640,360),interpolation=cv2.INTER_AREA))]:
                sx,sy=im.shape[1]/image.shape[1],im.shape[0]/image.shape[0]
                boxes=[[v*(sx if n%2==0 else sy) for n,v in enumerate(t['bbox'])] for t in row['objects']]
                out['manual'][res]=manual_result(im,boxes,backend,fid,('PHOTO_MANUAL',row['id'],res))
                observed_score(group,res+'_',out['manual'][res],row['objects'])
                overlay=im.copy()
                for i,(truth,box) in enumerate(zip(row['objects'],boxes)):
                    roi=crop(im,box);attr=asdict(backend.classify(roi));score_attr(group,res+'_',truth,attr)
                    x,y,X,Y=map(round,box);cv2.rectangle(overlay,(x,y),(X,Y),(0,220,255),1)
                    text(overlay,f"#{i} {attr['color']} {attr['shape']}",(x,max(14,y-4)),.4)
                    tile=np.zeros((230,460,3),np.uint8)
                    for n,line in enumerate([f"{row['id']} #{i} {res} {roi.shape[1]}x{roi.shape[0]}",
                        'GT '+truth['color']+' '+truth['shape'],attr['color']+' '+attr['shape'],attr['reason']]):text(tile,line,(5,18+21*n),.38)
                    scale=min(440/max(1,roi.shape[1]),130/max(1,roi.shape[0]));r=cv2.resize(roi,(max(1,round(roi.shape[1]*scale)),max(1,round(roi.shape[0]*scale))),interpolation=cv2.INTER_NEAREST)
                    tile[95:95+r.shape[0],:r.shape[1]]=r
                    if row['split']=='NEW_SCENE_REVIEWED_BEFORE_MODEL_RUN':alltiles.append(tile)
                    if attr['color']!=truth['color'] or attr['shape']!=truth['shape']:
                        stem=f"{row['id']}_{i}_{res}";cv2.imwrite(str(failures/(stem+'.png')),roi);dump(failures/(stem+'.json'),dict(truth=truth,predicted=attr,path=str(path)))
                cv2.imwrite(str(args.output/f"{row['id']}_{res}_manual.jpg"),overlay)
                # Independent full-image fresh inference, no truth supplied to detectors.
                out['detected'][res]={}
                for key,model in models.items():
                    detections,ms=model.detect(im)
                    tracker=MultiSignalTracker(backend=backend,confirm_updates=1)
                    ns=1_000_000_000+fid*1_000_000_000
                    tracker.update(im,[Candidate(tuple(d['bbox']),d['confidence'],fid,ns,key) for d in detections if d['confidence']>=.25],frame_id=fid,timestamp_ns=ns,context=('PHOTO_DETECTED',row['id'],res,key))
                    result=associate(tracker.snapshot(ns),now_ns=ns)
                    out['detected'][res][key]=dict(raw=detections,result=result,inference_ms=ms)
                    obs=im.copy()
                    for s in result['signals']:
                        x,y,X,Y=map(round,s['bbox']);cv2.rectangle(obs,(x,y),(X,Y),(0,210,255),1);text(obs,f"#{s['track_id']} {s['color']} {s['shape']}",(x,max(14,y-4)),.4)
                    cv2.imwrite(str(args.output/f"{row['id']}_{res}_{key}.jpg"),obs)
            stream.write(json.dumps(out,ensure_ascii=False,allow_nan=False)+'\n')
    while len(alltiles)%4:alltiles.append(np.zeros_like(alltiles[0]))
    cv2.imwrite(str(args.output/'new_scene_roi_comparison.jpg'),np.vstack([np.hstack(alltiles[n:n+4]) for n in range(0,len(alltiles),4)]))
    report={split:dict(counts=dict(c),accuracy={k[:-6]:dict(correct=c[k[:-6]+'_correct'],total=v,accuracy=c[k[:-6]+'_correct']/v) for k,v in c.items() if k.endswith('_total') and v}) for split,c in counts.items()}
    dump(args.output/'report.json',dict(splits=report,images=len(rows),scope='MANUAL_ROI_ATTRIBUTES_SEPARATE_FROM_FRESH_DETECTION; partial labels do not establish detector precision or ego-road accuracy'))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
