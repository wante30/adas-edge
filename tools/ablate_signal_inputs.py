"""Offline diagnostic ablations; does not change deployed inference scheduling."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.offline_signal_models import OfflineDetector
from tools.report_signal_diagnostics import lamp_matches
from tools.evaluate_signals_offline import dump,score_attr
from competition_ai import multi_signal,signal_shape


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/'docs/multi_signal/offline_eval_manifest.json').read_text(encoding='utf-8'))
    models={k:OfflineDetector(Path(manifest['model_root'])/v,traffic_light_id=8) for k,v in dict(
        bdd='rknn_b05o/bdd_yolov8n_512x288_raw_heads.onnx',p2='rknn_b05o/b05n_p2_multiclass_512x288_raw_heads_20261026.onnx').items()}
    counts,rows={},[]
    for clip in manifest['videos'][:2]:
        c=cv2.VideoCapture(clip['path'])
        for gt in clip['labels']:
            c.set(1,gt['frame_id']);ok,im=c.read()
            if not ok:raise ValueError('Frame unavailable')
            q=cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA)
            for model,detector in models.items():
                for name,image,scale,offset in [('q4_full',q,1,0),('q4_top60',q[:216],1,0),
                    ('q4_center60_zoom',q[:216,128:512],1,128),('original_full',im,3,0),('original_top60',im[:648],3,0)]:
                    detections,ms=detector.detect(image)
                    for d in detections:d['bbox']=[v/scale+(offset if n%2==0 else 0) for n,v in enumerate(d['bbox'])]
                    matches=lamp_matches(detections,gt['objects'],3,3)
                    key=clip['id']+'_'+model+'_'+name;counter=counts.setdefault(key,Counter())
                    counter['correct']+=len(matches);counter['total']+=len(gt['objects'])
                    rows.append(dict(clip=clip['id'],frame_id=gt['frame_id'],model=model,input_mode=name,
                                     matched=list(matches),raw=detections,ms=ms))
        c.release()
    dump(args.output/'detection_report.json',counts)
    detection_counts=counts
    (args.output/'detections.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    dump(args.output/'models.json',{k:m.evidence for k,m in models.items()})
    labels=json.loads((ROOT/'docs/multi_signal/photo_labels.json').read_text(encoding='utf-8'))['images']
    caps={}
    original_bound=multi_signal.bounded_roi
    try:
        for edge in (192,384,768):
            def bound(roi):
                h,w=roi.shape[:2]
                if max(h,w)<=edge:return roi
                s=edge/max(h,w)
                return cv2.resize(roi,(max(1,round(w*s)),max(1,round(h*s))),interpolation=cv2.INTER_AREA)
            multi_signal.bounded_roi=signal_shape.bounded_roi=bound
            backend=signal_shape.ContourTemplateBackend();counts=Counter()
            for row in labels:
                im=cv2.imread(str(ROOT/'outputs/real_signal_photos'/(row['id']+'.jpg')))
                for truth in row['objects']:
                    x,y,X,Y=truth['bbox'];a=backend.classify(im[y:Y,x:X])
                    score_attr(counts,'',truth,dict(color=a.color,shape=a.shape))
            caps[str(edge)]=dict(counts)
    finally:multi_signal.bounded_roi=signal_shape.bounded_roi=original_bound
    dump(args.output/'roi_cap_report.json',caps)
    print(json.dumps(dict(detector=detection_counts,caps=caps),indent=2))


if __name__=='__main__':main()
