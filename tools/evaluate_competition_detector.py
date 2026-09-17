"""Run CompetitionDetector over supplied videos; requires RKNN Lite on target."""
from __future__ import annotations
import argparse, csv, json, statistics
from pathlib import Path
import cv2
from competition_ai import CompetitionDetector
from competition_ai.input_shape import validate_model_shape

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--input',required=True); p.add_argument('--input-size',default='320',help='static canvas: 320 or WxH, e.g. 512x288'); p.add_argument('--output-dir',required=True); p.add_argument('--conf',type=float,default=.25); p.add_argument('--max-frames',type=int); a=p.parse_args(); a.input_size=validate_model_shape(a.input_size)
    paths=[Path(a.input)] if Path(a.input).is_file() else sorted(Path(a.input).glob('*.mp4'))
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); det_rows=[]; frame_rows=[]; counts={}; lat=[]; total=0
    with CompetitionDetector(a.model,a.input_size,a.conf) as detector:
      for video in paths:
        cap=cv2.VideoCapture(str(video)); writer=None; fid=0
        while cap.isOpened() and (a.max_frames is None or fid<a.max_frames):
          ok,frame=cap.read()
          if not ok: break
          if writer is None: writer=cv2.VideoWriter(str(out/f'{video.stem}_annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),cap.get(cv2.CAP_PROP_FPS) or 25,(frame.shape[1],frame.shape[0]))
          r=detector.infer(frame,0,fid); lat.append(r.inference_ms); total+=1; frame_rows.append({'video':video.name,'frame_id':fid,'detections':len(r.detections),'inference_ms':r.inference_ms})
          for d in r.detections:
            counts[d.class_name]=counts.get(d.class_name,0)+1; det_rows.append({'video':video.name,'frame_id':fid,'timestamp':fid/(cap.get(cv2.CAP_PROP_FPS) or 25),'class':d.class_name,'confidence':d.confidence,'x1':d.x1,'y1':d.y1,'x2':d.x2,'y2':d.y2,'inference_ms':r.inference_ms}); cv2.rectangle(frame,(int(d.x1),int(d.y1)),(int(d.x2),int(d.y2)),(0,255,0),2)
          writer.write(frame); fid+=1
        cap.release(); writer and writer.release()
    summary={'total_frames':total,'total_detections':sum(counts.values()),'detections_per_class':counts,'empty_frame_ratio':sum(not x['detections'] for x in frame_rows)/total if total else 0,'average_detections_per_frame':sum(x['detections'] for x in frame_rows)/total if total else 0,'inference_latency_ms':{k:getattr(statistics,'median' if k=='p50' else 'mean')(lat) if lat else 0 for k in ('mean','p50') }|{'p95':float(__import__('numpy').percentile(lat,95)) if lat else 0}}
    for name,rows in [('detections.csv',det_rows),('frame_stats.csv',frame_rows)]:
      with (out/name).open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else (['video','frame_id','detections','inference_ms'] if name.startswith('frame') else [])); w.writeheader(); w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); (out/'summary.md').write_text('# Competition detector evaluation\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n',encoding='utf-8')
if __name__=='__main__': main()
