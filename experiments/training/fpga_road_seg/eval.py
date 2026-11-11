import argparse
from torch.utils.data import DataLoader
from .dataset_adapter import RoadDataset, write_json, file_hash
from .train import load_checkpoint, names_for
from .engine import evaluate


def main():
    p=argparse.ArgumentParser(description='Evaluate held-out masks, respecting per-channel validity')
    p.add_argument('--checkpoint',required=True); p.add_argument('--manifest',required=True)
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--output',required=True); p.add_argument('--device',default='cpu')
    args=p.parse_args()
    model,ck=load_checkpoint(args.checkpoint,args.device)
    cfg=ck['config']; baseline=cfg['model'] in ('twin-small','twin-nano')
    ds=RoadDataset(args.manifest,args.split,(cfg['width'],cfg['height']),baseline)
    report=evaluate(model,DataLoader(ds,batch_size=cfg['batch_size']),args.device,names_for(cfg['model']),cfg.get('threshold',.5))
    report.update(split=args.split,evidence=ck['evidence'],checkpoint_sha256=file_hash(args.checkpoint),manifest_sha256=file_hash(args.manifest))
    write_json(args.output,report)


if __name__ == '__main__':
    main()
