"""Clearly synthetic fixtures, never evidence of road accuracy."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .dataset_adapter import validate_manifest, write_json


def generate(output, count=24, size=(320,192), seed=15):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(seed); w,h=size; rows=[]
    for i in range(count):
        y=np.zeros((4,h,w),np.uint8)
        shift=int(rng.integers(-15,16))
        cv2.fillPoly(y[0],[np.array([[w//2-30+shift,h//3],[w//2+30+shift,h//3],[w-10,h-1],[10,h-1]])],1)
        cv2.line(y[1],(w//2-25+shift,h//3),(w//4,h-1),1,3)
        for j in range(3):
            yy=h//2+j*28
            cv2.line(y[2],(w//2+20+shift+j*8,yy),(w//2+24+shift+j*8,yy+14),1,3)
        for j in range(5):
            cv2.rectangle(y[3],(w//3+j*20,h-40),(w//3+8+j*20,h-25),1,-1)
        image=np.clip(rng.normal(35,4,(h,w,3)),0,255).astype(np.uint8)
        for c,color in enumerate(((70,90,60),(240,210,20),(50,180,250),(230,100,200))):
            image[y[c]>0]=color
        ip=output/f'{i:04d}.png'; mp=output/f'{i:04d}.npz'
        Image.fromarray(image).save(ip)
        np.savez_compressed(mp,target=y,valid=np.ones_like(y),lane=y[1]|y[2],lane_valid=np.ones((h,w),np.uint8))
        rows.append(dict(id=f'synthetic-{i}',image=str(ip),masks=str(mp),source='synthetic',
                         split='train' if i<count*2//3 else 'val' if i<count*5//6 else 'test',
                         group_id=f'synthetic-independent-scene-{i}',license='generated fixture'))
    validate_manifest(rows)
    manifest=output/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    write_json(output/'README.json',dict(evidence='synthetic_smoke',seed=seed,count=count,size=size))
    return manifest


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    args=p.parse_args(); print(generate(args.output))


if __name__ == '__main__':
    main()
