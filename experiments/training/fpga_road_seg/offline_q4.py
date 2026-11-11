"""Offline FAST-Q4 adapter: mock masks + same-frame detection JSON + overlays.

No import from, or changes to, live FAST-Q4, PCIe, HSST, HUD or BEV paths.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from . import CLASSES
from .dataset_adapter import read_json,write_json,file_hash
from .infer import overlay
from .mask_protocol import encode,OfflineMaskProvider,transfer_budget


def crop_q4(image, layout):
    if layout=='q4_native': return image
    if layout!='mosaic_2x2' or image.shape[0]%2 or image.shape[1]%2:
        raise ValueError('explicit even-sized 2x2 mosaic required')
    h,w=image.shape[:2]
    return image[h//2:,w//2:]  # Q4 = zero-based quadrant 3, never Q1 road default.


def validate_detections(detections, *, frame_id, timestamp_ns, shape, stream_epoch=0):
    expected=dict(frame_id=frame_id,timestamp_ns=timestamp_ns,shape=list(shape),stream_epoch=stream_epoch,
                  source_id='Q4',coordinate_space='q4_native')
    if any(detections.get(k)!=v for k,v in expected.items()):
        raise ValueError('detection frame/timestamp/epoch/geometry/coordinate mismatch')
    h,w=shape
    for row in detections['detections']:
        box=row['bbox']
        if len(box)!=4 or not all(type(n) in (int,float) and np.isfinite(n) for n in box):
            raise ValueError('invalid detection bbox')
        x1,y1,x2,y2=box
        if not 0<=x1<x2<=w or not 0<=y1<y2<=h:
            raise ValueError('bbox outside Q4-native pixels')
    return detections['detections']


def fuse(mask, rows):
    """Descriptive lower-bbox overlap, not a safety/vehicle control decision."""
    result=[]
    for row in rows:
        x1,y1,x2,y2=row['bbox']
        # Bottom 20% of bounding box is an illustrative footprint approximation.
        left,top=int(np.floor(x1)),int(np.floor(y1+.8*(y2-y1)))
        right,bottom=int(np.ceil(x2)),int(np.ceil(y2))
        footprint=mask[:,top:bottom,left:right]>=.5
        result.append(dict(row,mask_footprint_overlap={n:float(v.mean()) for n,v in zip(CLASSES,footprint)},
                           fusion='offline descriptive overlap; no physical distance/control claim'))
    return result


def run(image_path, mask_path, detections_path, output, *, frame_id, timestamp_ns, now_ns,
        layout='q4_native',encoding='bitpack-le',stream_epoch=0,max_age_ms=150):
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    source=np.asarray(Image.open(image_path).convert('RGB'))
    image=crop_q4(source,layout)
    mask=np.load(mask_path,allow_pickle=False)
    if mask.shape!=(4,*image.shape[:2]):
        raise ValueError('mask must match cropped Q4-native image, not mosaic/network padding')
    packet=encode(mask,frame_id=frame_id,timestamp_ns=timestamp_ns,stream_epoch=stream_epoch,encoding=encoding)
    provider=OfflineMaskProvider(packet,max_age_ms)
    usable,status=provider.get(frame_id=frame_id,timestamp_ns=timestamp_ns,stream_epoch=stream_epoch,
                               now_ns=now_ns,shape=image.shape[:2])
    rows=validate_detections(read_json(detections_path),frame_id=frame_id,timestamp_ns=timestamp_ns,
                             stream_epoch=stream_epoch,shape=image.shape[:2])
    fused=fuse(usable,rows) if usable is not None else []
    shown=overlay(image,usable,CLASSES,'MOCK MASK + OFFLINE DETECTIONS / NO HARDWARE') if usable is not None else Image.fromarray(image)
    draw=ImageDraw.Draw(shown)
    if usable is None: draw.text((4,4),'MOCK MASK REJECTED: STALE/INVALID',fill='red')
    for row in rows:
        draw.rectangle(row['bbox'],outline='white',width=2)
        draw.text((row['bbox'][0],row['bbox'][1]),str(row.get('label','object')),fill='white')
    shown.save(out/'overlay.png')
    write_json(out/'mask-packet.json',packet)
    report=dict(evidence='offline_mock_only',status=status,frame_id=frame_id,timestamp_ns=timestamp_ns,
                source_id='Q4',coordinate_space='q4_native',shape=list(image.shape[:2]),
                input_layout=layout,detections=rows,fused=fused,image_sha256=file_hash(image_path),
                mask_sha256=file_hash(mask_path),detections_sha256=file_hash(detections_path),
                transfer_budget=transfer_budget(image.shape[1],image.shape[0]))
    write_json(out/'fusion-report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True); p.add_argument('--mask-npy',required=True)
    p.add_argument('--detections',required=True); p.add_argument('--output',required=True)
    p.add_argument('--frame-id',type=int,required=True); p.add_argument('--timestamp-ns',type=int,required=True)
    p.add_argument('--now-ns',type=int,required=True); p.add_argument('--stream-epoch',type=int,default=0)
    p.add_argument('--layout',choices=['q4_native','mosaic_2x2'],default='q4_native')
    p.add_argument('--encoding',choices=['bitpack-le','rle-u32','probability-u8'],default='bitpack-le')
    p.add_argument('--max-age-ms',type=float,default=150)
    a=p.parse_args()
    print(run(a.image,a.mask_npy,a.detections,a.output,frame_id=a.frame_id,timestamp_ns=a.timestamp_ns,
              now_ns=a.now_ns,stream_epoch=a.stream_epoch,layout=a.layout,encoding=a.encoding,max_age_ms=a.max_age_ms)['status'])


if __name__=='__main__': main()
