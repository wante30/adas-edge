"""Check delivered artifact identity, task validity, metrics and video readability."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import onnx

from .dataset_adapter import file_hash, read_json, write_json


def verify(output):
    out=Path(output)
    required=['README_RESULT.md','metrics.csv','training_curves.png','FPGA_HANDOFF.md',
              'evaluation.json','exports/search.json','exports/comparison.json',
              'exports/model.fp32.onnx','exports/model.int8.qdq.onnx','exports/quantization.json',
              'checkpoints/candidate-plus/best.pt','comparisons/video.json',
              'comparisons/real_video_before_after.mp4','failures/crosswalk_errors.json','run.log']
    for name in required:
        if not (out/name).is_file() or not (out/name).stat().st_size:
            raise ValueError(f'missing/empty delivered artifact: {name}')
    quant=read_json(out/'exports/search.json');evaluation=read_json(out/'evaluation.json')
    checkpoint_hash=file_hash(out/'checkpoints/candidate-plus/best.pt')
    if checkpoint_hash!=quant['checkpoint_sha256'] or checkpoint_hash!=evaluation['checkpoint_sha256']:
        raise ValueError('checkpoint/evaluation/ONNX export provenance mismatch')
    for name in ('model.fp32.onnx','model.int8.qdq.onnx'):
        model=onnx.load(str(out/'exports'/name));onnx.checker.check_model(model)
        shape=lambda tensor:[d.dim_value for d in tensor.type.tensor_type.shape.dim]
        if shape(model.graph.input[0])!=[1,3,192,320] or shape(model.graph.output[0])!=[1,4,192,320]:
            raise ValueError(f'ONNX shape contract mismatch: {name}')
    selected=next(r for r in quant['variants'] if r['name']==quant['selected'])
    if file_hash(out/'exports/model.int8.qdq.onnx')!=selected['sha256']:
        raise ValueError('selected INT8 file differs from measured variant')
    metrics=list(csv.DictReader((out/'metrics.csv').open(encoding='utf-8')))
    if len(metrics)!=32:
        raise ValueError(f'expected 4 variants x 2 domains x 4 channels, got {len(metrics)}')
    sources={'before_fp32':evaluation['before'],'after_fp32':evaluation['after'],
             'after_onnx_fp32':quant['comparison']['fp32'],'after_int8':quant['comparison']['int8']}
    identities=set()
    for row in metrics:
        key=row['variant'],row['dataset'],row['class_name']
        if key in identities:raise ValueError(f'duplicate metric row: {key}')
        identities.add(key)
        truth=sources[row['variant']][row['dataset']]['classes'][row['class_name']]
        for metric in ('iou','f1','recall','valid_pixels','tp','fp','fn','tn'):
            actual=None if row[metric]=='' else float(row[metric])
            if actual is None and truth[metric] is None:continue
            if actual is None or truth[metric] is None or not np.isclose(actual,truth[metric],rtol=1e-10,atol=1e-12):
                raise ValueError(f'CSV/result mismatch: {key} {metric}')
        expected_samples=10000 if row['dataset']=='bdd100k' else 439
        if int(row['samples'])!=expected_samples:
            raise ValueError(f'wrong evaluation population: {key}')
        if (row['dataset'],row['class_name']) in [('rlmd','drivable_area'),('bdd100k','crosswalk')]:
            if int(row['valid_pixels']) or row['iou'] or row['f1'] or row['recall']:
                raise ValueError(f'missing label was scored as background: {key}')
    video=read_json(out/'comparisons/video.json')
    cap=cv2.VideoCapture(str(out/'comparisons/real_video_before_after.mp4'))
    try:
        if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=video['frames']:
            raise ValueError('video unreadable or frame count mismatch')
        for index in (0,video['frames']//2,video['frames']-1):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=cap.read()
            if not ok or frame.shape[:2]!=(360,1920):raise ValueError('video decode/shape mismatch')
    finally:
        cap.release()
    record=dict(status='verified',metric_rows=len(metrics),checkpoint_sha256=checkpoint_hash,
                selected_quantizer=quant['selected'],onnx_shapes='fixed [1,3,192,320] -> [1,4,192,320]',
                video_decoded_frames=[0,video['frames']//2,video['frames']-1],
                missing_tasks='RLMD drivable and BDD crosswalk have zero valid pixels and null scores',
                artifacts=[dict(path=name,bytes=(out/name).stat().st_size,sha256=file_hash(out/name)) for name in required])
    write_json(out/'artifact_verification.json',record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    args=p.parse_args();print(verify(args.output)['status'])
