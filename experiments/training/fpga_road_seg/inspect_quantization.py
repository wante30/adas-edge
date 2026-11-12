"""Real-image FP32/INT8 disagreement gallery and measured CPU ORT latency."""
import argparse
import hashlib
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch

from . import CLASSES
from .dataset_adapter import read_index, read_json, letterbox, restore, write_json
from .export import session
from .infer import overlay
from .quantize_stage2 import maximum_drop


def inspect(output, manifest):
    out=Path(output);quant=read_json(out/'exports/search.json')
    fp=session(out/'exports/model.fp32.onnx');iq=session(out/'exports/model.int8.qdq.onnx')
    rows=read_index(manifest)
    folder=out/'comparisons/quantization';folder.mkdir(exist_ok=True)
    picked=[]
    for source,split in [('bdd100k','test'),('rlmd','val')]:
        candidates=sorted([r for r in rows if r['source']==source and r['split']==split],
                          key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())
        if source=='rlmd':
            picked.extend([r for r in candidates if r['positive_pixels'][3]>0][:2])
            picked.extend([r for r in candidates if r['positive_pixels'][3]==0][:2])
        else:picked.extend(candidates[:3])
    gallery=[]
    for row in picked:
        image=np.asarray(Image.open(row['image']).convert('RGB'));x,g=letterbox(image)
        a=fp.run(None,{'rgb':x[None].numpy()})[0];b=iq.run(None,{'rgb':x[None].numpy()})[0]
        pa,pb=torch.from_numpy(a).sigmoid().numpy()[0],torch.from_numpy(b).sigmoid().numpy()[0]
        with np.load(row['masks']) as m:truth=m['target'].copy()
        for c in row.get('disabled_channels',[]):truth[CLASSES.index(c)]=0
        disagreement=(restore(pa,g)>=.5)!=(restore(pb,g)>=.5)
        error=np.zeros_like(image,dtype=np.float32)
        for c,color in enumerate([(0,210,100),(255,190,0),(0,160,255),(235,70,220)]):
            error[disagreement[c]]=color
        error=Image.fromarray(error.astype(np.uint8))
        panels=[('Original '+row['id'],Image.fromarray(image)),
                ('GT (missing channels hidden)',overlay(image,truth,CLASSES,'GT')),
                ('FP32 all four predictions',overlay(image,restore(pa,g),CLASSES,'FP32')),
                ('INT8 all four predictions',overlay(image,restore(pb,g),CLASSES,'INT8')),
                ('Threshold disagreement (not ground-truth error)',error)]
        canvas=Image.new('RGB',(1920,772),'white');draw=ImageDraw.Draw(canvas)
        for i,(label,im) in enumerate(panels):
            x0,y0=i%3*640,i//3*386
            draw.text((x0+4,y0+4),label,fill='black');canvas.paste(im.resize((640,360)),(x0,y0+26))
        canvas.save(folder/(row['id']+'.jpg'),quality=90)
        gallery.append(dict(id=row['id'],source=row['source'],split=row['split'],
                            threshold_disagreement_pixels_native_per_channel=disagreement.sum((1,2)).tolist()))
    # Same actual road input and CPU execution provider/thread count for both.
    image=np.asarray(Image.open(picked[0]['image']).convert('RGB'));x,_=letterbox(image)
    inputs={'rgb':x[None].numpy()};timing={}
    for name,runtime in [('fp32',fp),('int8',iq)]:
        for _ in range(20):runtime.run(None,inputs)
        samples=[]
        for _ in range(100):
            start=time.perf_counter();runtime.run(None,inputs);samples.append((time.perf_counter()-start)*1000)
        timing[name]=dict(p50_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)))
    write_json(out/'quantization_inspection.json',dict(selected=quant['selected'],gallery=gallery,cpu_ort_latency=timing,
        latency_scope='ONNX Runtime CPUExecutionProvider, 4 intra-op threads, batch1, 20 warmups/100 runs, preprocessed real road input; excludes I/O/preprocessing; not GPU/FPGA speed',
        full_comparison_maximum_key_metric_drop=maximum_drop(quant['comparison']['fp32'],quant['comparison']['int8'])))
    labels=['BDD drivable IoU','BDD solid F1','BDD dashed F1','RLMD crosswalk IoU']
    keys=[('bdd100k','drivable_area','iou'),('bdd100k','solid_lane','f1'),('bdd100k','dashed_lane','f1'),('rlmd','crosswalk','iou')]
    variants=[dict(name='FP32',domains=quant['fp32_validation'])]+quant['variants']
    fig,ax=plt.subplots(figsize=(12,5));width=.8/len(variants);indices=np.arange(len(keys))
    for i,v in enumerate(variants):
        values=[100*v['domains'][s]['classes'][c][m] for s,c,m in keys]
        ax.bar(indices-.4+(i+.5)*width,values,width,label=v['name'])
    ax.set_xticks(indices,labels);ax.set_ylabel('Internal validation (%)');ax.set_ylim(0,75)
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(out/'quantization_comparison.png',dpi=150);plt.close(fig)
    history_file=out/'exports/qat-history.json'
    if history_file.exists():
        history=read_json(history_file);fig,axes=plt.subplots(1,2,figsize=(11,4))
        epochs=[e['epoch'] for e in history]
        axes[0].plot(epochs,[e['loss'] for e in history],marker='o',label='QAT supervised + masked FP32 regularizer')
        for source,cls,metric in keys:
            axes[1].plot(epochs,[e['domains'][source]['classes'][cls][metric] for e in history],marker='o',label=source+' '+cls+' '+metric)
        for ax in axes:ax.set_xlabel('QAT epoch');ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/'qat_training_curves.png',dpi=150);plt.close(fig)
    return timing


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--manifest',required=True)
    args=p.parse_args();print(inspect(args.output,args.manifest))
