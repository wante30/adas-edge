import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .dataset_adapter import RoadDataset, write_json, file_hash
from .train import load_checkpoint, names_for
from .engine import evaluate


def main():
    p=argparse.ArgumentParser(description='Evaluate held-out masks, respecting per-channel validity')
    p.add_argument('--checkpoint',required=True); p.add_argument('--manifest',required=True)
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--output',required=True); p.add_argument('--device',default='cpu')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    torch.set_num_threads(4)
    model,ck=load_checkpoint(args.checkpoint,args.device)
    cfg=ck['config']; baseline=cfg['model'] in ('twin-small','twin-nano')
    ds=RoadDataset(args.manifest,args.split,(cfg['width'],cfg['height']),baseline,ck.get('excluded_channels', ()))
    report=evaluate(model,DataLoader(ds,batch_size=cfg['batch_size'],num_workers=args.workers,
                    pin_memory=args.device.startswith('cuda')),args.device,names_for(cfg['model']),cfg.get('threshold',.5))
    report.update(split=args.split,evidence=ck['evidence'],checkpoint_sha256=file_hash(args.checkpoint),manifest_sha256=file_hash(args.manifest))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)


if __name__ == '__main__':
    main()
