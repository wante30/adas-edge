"""Verify official state_dicts, original 640x384 preprocessing and FP32 export.

No segmentation GT is available here: inference/parity only, never paper accuracy.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import onnx

from .dataset_adapter import letterbox, restore, write_json, file_hash
from .infer import overlay
from .export import session, inventory
from .vendor.twinlite.model import TwinLiteNetPlus


def run(weights, images, variant, output):
    torch.set_num_threads(4)
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    model=TwinLiteNetPlus(SimpleNamespace(config=variant)).eval()
    state=torch.load(weights,map_location='cpu',weights_only=True)
    # Downloads use state_dict only; fail on mismatch rather than silently discard keys.
    model.load_state_dict(state,strict=True)
    target=out/f'twin-{variant}-original.fp32.onnx'
    with torch.no_grad():
        torch.onnx.export(model,torch.zeros(1,3,384,640),str(target),input_names=['rgb'],
                          output_names=['drivable_2class_logits','lane_2class_logits'],opset_version=17,
                          do_constant_folding=True,dynamic_axes=None)
    onnx.checker.check_model(onnx.load(str(target)))
    ort_model=session(target); results=[]
    paths=sorted(Path(images).glob('*.jpg'))
    if not paths:
        raise ValueError('expected upstream demo JPG images')
    for path in paths:
        image=np.asarray(Image.open(path).convert('RGB'))
        x,g=letterbox(image,size=(640,384))
        if image.shape[:2]!=(720,1280) or g['top']!=12:
            raise ValueError('original recipe probe requires 1280x720 images -> 640x360 + 12px padding')
        with torch.no_grad():
            outputs=model(x[None]); reference=[t.numpy() for t in outputs]
            prob=np.stack([t.softmax(1)[0,1].numpy() for t in outputs])
        exported=ort_model.run(None,{'rgb':x[None].numpy()})
        error=max(float(np.max(np.abs(a-b))) for a,b in zip(reference,exported))
        if error>1e-3:
            raise RuntimeError(f'official checkpoint FP32 ONNX parity failure: {error}')
        native_prob=restore(prob,g)
        overlay(image,native_prob,('drivable_area','lane'),f'OFFICIAL {variant.upper()} / NO GT / SOFTWARE ONLY').save(out/f'{path.stem}.png')
        results.append(dict(image=path.name,image_sha256=file_hash(path),geometry=g,
                            torch_onnx_max_logit_error=error,
                            positive_native_pixels=[int((p>=.5).sum()) for p in native_prob]))
    report=dict(evidence='official_pretrained_inference_only_no_GT',variant=variant,
                upstream_commit='90f1b8695ae311d5123b05f8534b2e11e42499d2',weights_sha256=file_hash(weights),
                weights_bytes=Path(weights).stat().st_size,parameters=sum(p.numel() for p in model.parameters()),
                input_shape=[1,3,384,640],output_shapes=[[1,2,384,640],[1,2,384,640]],
                evaluation_crop='[:,:,12:-12,:] at original 640x384; same geometry as upstream utils.val',
                fp32_onnx_sha256=file_hash(target),samples=results,operators=inventory(target)['operators'],
                accuracy='not measured; no labelled evaluation set',hardware='not run')
    write_json(out/'reference-report.json',report)
    write_json(out/'operators.json',inventory(target))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',required=True); p.add_argument('--images',required=True)
    p.add_argument('--variant',choices=['small','nano'],default='small'); p.add_argument('--output',required=True)
    args=p.parse_args(); r=run(args.weights,args.images,args.variant,args.output)
    print(r['variant'],r['parameters'],'images',len(r['samples']),'max_error',max(s['torch_onnx_max_logit_error'] for s in r['samples']))


if __name__=='__main__': main()
