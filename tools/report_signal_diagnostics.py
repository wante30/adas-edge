"""Review fresh inference with independently drawn head and lamp ground truth.

Input must be outputs of this session's evaluate_signals_offline.py, never the
old COCO audit cache. Produces explicit alternative localization metrics, not
a replacement for head-IoU recall. All labeled frames retain their denominators.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_signals_offline import load_baseline, export_baseline, crop, dump, score_attr, text, matches
from competition_ai.multi_signal import iou, Candidate, MultiSignalTracker
from competition_ai.signal_shape import ContourTemplateBackend
from competition_ai.signal_association import associate, ARROWS


def lamp_matches(rows, truths, sx, sy, threshold=.25):
    edges = []
    for ri, row in enumerate(rows):
        if row['confidence'] < threshold:
            continue
        a = [v*(sx if n%2 == 0 else sy) for n,v in enumerate(row['bbox'])]
        for ti, truth in enumerate(truths):
            lamp, head = truth['lamp_bbox_native'], truth['bbox_native']
            inter = max(0,min(a[2],lamp[2])-max(a[0],lamp[0]))*max(0,min(a[3],lamp[3])-max(a[1],lamp[1]))
            lamp_area = (lamp[2]-lamp[0])*(lamp[3]-lamp[1])
            area = (a[2]-a[0])*(a[3]-a[1])
            head_area = (head[2]-head[0])*(head[3]-head[1])
            cx, cy = (a[0]+a[2])/2, (a[1]+a[3])/2
            if (inter/lamp_area >= .5 and .25*lamp_area <= area <= 1.5*head_area
                    and head[0] <= cx <= head[2] and head[1] <= cy <= head[3]):
                edges.append((inter/lamp_area, ri, ti))
    used_r, used_t, found = set(), set(), {}
    for coverage, ri, ti in sorted(edges, reverse=True):
        if ri not in used_r and ti not in used_t:
            found[ti] = ri; used_r.add(ri); used_t.add(ti)
    return found


def observed_score(counts, prefix, result, truths):
    for direction in ('left', 'straight', 'right'):
        wanted = {t['color'] for t in truths if ARROWS.get(t['shape']) == direction and t['color'] != 'UNKNOWN'}
        target = next(iter(wanted)) if len(wanted) == 1 else 'UNKNOWN'
        output = result['observed_directions'][direction]['color']
        kind = 'positive' if target != 'UNKNOWN' else 'unsupported'
        counts[prefix+'observed_'+kind+'_total'] += 1
        counts[prefix+'observed_'+kind+'_correct'] += output == target
        counts[prefix+'ego_unknown_total'] += 1
        counts[prefix+'ego_unknown_correct'] += result['directions'][direction]['color'] == 'UNKNOWN'


def manual_result(image, boxes, backend, fid, context):
    tracker = MultiSignalTracker(backend=backend, confirm_updates=1, split_heads=False)
    ns = 1_000_000_000+fid*1_000_000_000
    tracker.update(image,[Candidate(tuple(b),1.,fid,ns,'MANUAL_ROI') for b in boxes],
                   frame_id=fid,timestamp_ns=ns,context=context)
    return associate(tracker.snapshot(ns),now_ns=ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest = json.loads((ROOT/'docs/multi_signal/offline_eval_manifest.json').read_text(encoding='utf-8'))
    baseline = load_baseline(export_baseline(args.output/'baseline',manifest['baseline_ref']))
    improved = ContourTemplateBackend()
    reports = {}
    for clip in manifest['videos']:
        path = args.input/clip['id']/'frames.jsonl'
        if not path.exists() or not clip['labels']:continue
        raw_frames = {r['frame_id']:r for r in map(json.loads,path.read_text(encoding='utf-8').splitlines())}
        if not (args.input/'models.json').exists():
            raise ValueError('Fresh inference model manifest required')
        out=args.output/clip['id'];out.mkdir();(out/'failures').mkdir()
        cap=cv2.VideoCapture(clip['path']);fps=cap.get(5)
        counts,diagnostics,tiles=Counter(),[],[]
        writer=cv2.VideoWriter(str(out/'full_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,1000))
        labelmap={r['frame_id']:r['objects'] for r in clip['labels']}
        fid=0
        with (out/'frames.jsonl').open('x',encoding='utf-8') as stream:
            while True:
                ok,im=cap.read()
                if not ok:break
                frame=raw_frames[fid];q4=cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA)
                sx,sy=im.shape[1]/640,im.shape[0]/360
                canvas=np.zeros((1000,1920,3),np.uint8)
                for col,model in enumerate(list(frame['raw_detections'])[:3]):
                    panel=q4.copy()
                    for row in frame['raw_detections'][model]:
                        if row['confidence']<.25:continue
                        x,y,X,Y=map(round,row['bbox']);cv2.rectangle(panel,(x,y),(X,Y),(0,220,255),1)
                        text(panel,f"{row['confidence']:.2f}",(x,max(12,y-3)),.3)
                    canvas[32:392,col*640:(col+1)*640]=panel
                    text(canvas,f'FRESH {model} f{fid} | head/lamp diagnostics separate',(col*640+5,22),.45)
                    for j,s in enumerate(frame['pipelines'][model]['q4']['signals'][:6]):
                        text(canvas,f"#{s['track_id']} {s['color']} {s['shape']} {s['reason']}",(col*640+5,420+j*22),.34)
                entries=[]
                if fid in labelmap:
                    truth=[dict(t,bbox_q4=[v/(sx if n%2==0 else sy) for n,v in enumerate(t['bbox_native'])]) for t in labelmap[fid]]
                    counts['gt_objects']+=len(truth);counts['reviewed_frames']+=1
                    for model,rows in frame['raw_detections'].items():
                        for threshold in (.05,.15,.25):
                            counts[f'{model}_head_iou50_at{threshold}']+=len(matches(rows,truth,threshold))
                            counts[f'{model}_lamp_coverage50_at{threshold}']+=len(lamp_matches(rows,truth,sx,sy,threshold))
                        oldtruth=[dict(t,bbox_q4=t['previous_q4_bbox']) for t in truth]
                        counts[model+'_previous_gt_iou50_at0.25']+=len(matches(rows,oldtruth))
                        found=lamp_matches(rows,truth,sx,sy)
                        for ti,t in enumerate(truth):
                            if ti not in found:
                                counts[model+'_misses']+=1
                                candidates=[r for r in rows if r['confidence']>=.001]
                                best=max(candidates,key=lambda r:iou(r['bbox'],t['bbox_q4']),default=None)
                                failure=dict(stage='DETECTION',model=model,truth=t,
                                    reason='NO_ACCEPTED_LAMP_COVERAGE_AT_0.25',best_candidate=best)
                                entries.append(failure)
                                stem=f'{fid}_{ti}_{model}_detection'
                                cv2.imwrite(str(out/'failures'/(stem+'.png')),crop(im,t['attribute_roi_native']))
                                dump(out/'failures'/(stem+'.json'),failure)
                            else:
                                row=rows[found[ti]]
                                for res,image,box in [('q4',q4,row['bbox']),('original',im,[v*(sx if n%2==0 else sy) for n,v in enumerate(row['bbox'])])]:
                                    attr=asdict(improved.classify(crop(image,box)))
                                    score_attr(counts,'detected_'+model+'_'+res+'_',t,attr)
                                    entries.append(dict(stage='DETECTED_ATTRIBUTES',model=model,resolution=res,truth=t,attributes=attr))
                    for res,image in [('q4',q4),('original',im)]:
                        boxes=[]
                        for ti,t in enumerate(truth):
                            native=t['attribute_roi_native'];box=native if res=='original' else [v/(sx if n%2==0 else sy) for n,v in enumerate(native)]
                            boxes.append(box);roi=crop(image,box)
                            for version,backend in [('baseline',baseline),('improved',improved)]:
                                try:a=asdict(backend.classify(roi))
                                except cv2.error:a=dict(color='UNKNOWN',shape='UNKNOWN',reason='BASELINE_OPENCV_ERROR')
                                score_attr(counts,f'manual_{res}_{version}_',t,a)
                                entries.append(dict(stage='MANUAL_ROI',resolution=res,version=version,truth=t,attributes=a))
                                if version=='improved':
                                    tile=np.zeros((220,460,3),np.uint8)
                                    for n,line in enumerate([f'{clip["id"]} f{fid} #{ti} {res}',t['color']+' '+t['shape']+' -> '+a['color']+' '+a['shape'],a['reason']]):text(tile,line,(5,18+20*n),.37)
                                    scale=min(440/roi.shape[1],145/roi.shape[0]);r=cv2.resize(roi,(round(roi.shape[1]*scale),round(roi.shape[0]*scale)),interpolation=cv2.INTER_NEAREST)
                                    tile[70:70+r.shape[0],:r.shape[1]]=r;tiles.append(tile)
                                    # Original and Q4 GT panels; only reviewed frames.
                                    col=ti%2;row=0 if res=='q4' else 1
                                    canvas[540+row*220:760+row*220,col*460:col*460+460]=tile
                                    if (a['color'],a['shape'])!=(t['color'],t['shape']):
                                        stem=f'{fid}_{ti}_{res}';cv2.imwrite(str(out/'failures'/(stem+'.png')),roi)
                                        dump(out/'failures'/(stem+'.json'),entries[-1])
                        result=manual_result(image,boxes,improved,fid,('GT_ATTRIBUTE_ONLY',clip['id'],res))
                        observed_score(counts,f'manual_{res}_',result,truth)
                        frame['manual_'+res+'_association']=result
                    diagnostics.append(dict(frame_id=fid,entries=entries))
                text(canvas,'Reviewed GT frames show Q4/native crops. Others: detector/tracker output only. No interpolated truth.',(950,600),.4)
                text(canvas,'Separate: visible arrow direction / ego-road relevance UNKNOWN.',(950,630),.45)
                text(canvas,'Fresh float inference in this run; never old COCO cache. No FPGA / PCIe / Qt operations.',(950,660),.4)
                writer.write(canvas)
                frame['stage_diagnostics']=entries
                stream.write(json.dumps(frame,ensure_ascii=False,allow_nan=False)+'\n');fid+=1
        cap.release();writer.release()
        while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
        if tiles:cv2.imwrite(str(out/'manual_roi_comparison.jpg'),np.vstack([np.hstack(tiles[n:n+4]) for n in range(0,len(tiles),4)]))
        report=dict(counts=dict(counts),frames=fid,source_sha256=clip.get('sha256'),
            accuracy={key[:-6]:dict(correct=counts[key[:-6]+'_correct'],total=value,accuracy=counts[key[:-6]+'_correct']/value)
                      for key,value in counts.items() if key.endswith('_total') and value},
            association_scope='Observed-arrow accuracy and unsupported ego-relevance rejection; positive ego-road association NOT evaluated',
            localization_rules=dict(head='IoU>=0.5',lamp='>=50% manual luminous face covered; center inside head; 0.25*lamp_area<=pred_area<=1.5*head_area; one-to-one'),
            scope='Sparse reviewed overhead heads; no full-scene precision claims')
        dump(out/'report.json',report);dump(out/'diagnostics.json',diagnostics);reports[clip['id']]=report
        print(clip['id'],report['accuracy'],flush=True)
    dump(args.output/'report.json',reports)


if __name__=='__main__':main()
