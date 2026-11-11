import argparse
import numpy as np
from PIL import Image, ImageDraw
import torch
from .dataset_adapter import letterbox, restore, write_json, file_hash
from .train import load_checkpoint, names_for

COLORS=np.array([[0,210,100],[255,190,0],[0,160,255],[235,70,220]],np.uint8)


def overlay(image, probabilities, names, title, threshold=.5):
    canvas=image.astype(np.float32).copy()
    # Overlay precedence only: drivable -> solid -> dashed -> crosswalk.
    # Binary training channels remain independent and can overlap.
    for i,prob in enumerate(probabilities):
        positive=prob>=threshold
        canvas[positive]=.6*canvas[positive]+.4*COLORS[i]
    im=Image.fromarray(canvas.astype(np.uint8))
    draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,im.width,18*(len(names)+1)),fill=(15,15,15))
    draw.text((4,2),title,fill='white')
    for i,name in enumerate(names):
        draw.text((4,20+18*i),name,fill=tuple(map(int,COLORS[i])))
    return im


def main():
    p=argparse.ArgumentParser(description='Real image visualization with explicit checkpoint provenance')
    p.add_argument('--checkpoint',required=True); p.add_argument('--image',required=True)
    p.add_argument('--output',required=True); p.add_argument('--device',default='cpu')
    args=p.parse_args()
    model,ck=load_checkpoint(args.checkpoint,args.device); model.eval()
    cfg=ck['config']; image=np.asarray(Image.open(args.image).convert('RGB'))
    x,g=letterbox(image,size=(cfg['width'],cfg['height']))
    with torch.no_grad():
        prob=restore(model(x[None].to(args.device)).sigmoid()[0].cpu().numpy(),g)
    output_names=names_for(cfg['model'])
    for channel in ck.get('excluded_channels', ()):
        if channel in output_names:
            prob[output_names.index(channel)]=0
    label='SYNTHETIC-TRAINED / NOT ACCURACY EVIDENCE' if ck['evidence']=='synthetic_smoke' else 'CANDIDATE / NOT FPGA'
    overlay(image,prob,names_for(cfg['model']),label,cfg.get('threshold',.5)).save(args.output)
    write_json(str(args.output)+'.json',dict(checkpoint_sha256=file_hash(args.checkpoint),image_sha256=file_hash(args.image),evidence=ck['evidence'],geometry=g))


if __name__ == '__main__':
    main()
