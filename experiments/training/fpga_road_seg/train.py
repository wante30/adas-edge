"""Train real manifests, or explicitly labelled synthetic smoke fixtures."""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import CLASSES
from .dataset_adapter import RoadDataset, read_json, write_json, file_hash
from .engine import evaluate, masked_loss
from .models import build_model


def names_for(model):
    return ('drivable_area', 'lane') if model in ('twin-small', 'twin-nano') else CLASSES


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint['config']['model']).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model, checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--synthetic-smoke', action='store_true', help='Label results as synthetic; never a real-data metric')
    args = p.parse_args()
    cfg = read_json(args.config)
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if cfg['epochs'] < 1 or cfg['batch_size'] < 1:
        p.error('epochs and batch_size must be positive')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(4)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    size = cfg['width'], cfg['height']
    baseline = cfg['model'] in ('twin-small', 'twin-nano')
    train = RoadDataset(args.manifest, 'train', size, baseline)
    val = RoadDataset(args.manifest, 'val', size, baseline)
    if any(r['source'] == 'synthetic' for r in train.rows+val.rows) and not args.synthetic_smoke:
        p.error('synthetic fixture requires --synthetic-smoke')
    channels = len(names_for(cfg['model']))
    coverage = np.zeros(channels, np.int64)
    positives = coverage.copy()
    for row in train.rows:
        with np.load(row['masks'], allow_pickle=False) as masks:
            if baseline:
                y = np.stack([masks['target'][0], masks['lane']])
                v = np.stack([masks['valid'][0], masks['lane_valid']])
            else:
                y, v = masks['target'], masks['valid']
            coverage += v.sum((1,2)).astype(np.int64)
            positives += (y*v).sum((1,2)).astype(np.int64)
    # Refuse a four-class claim with absent channels/positive labels.
    if not args.synthetic_smoke and (np.any(coverage == 0) or np.any(positives == 0)):
        p.error(f'missing supervised training channels/positives: coverage={coverage}, positives={positives}')
    loader = DataLoader(train, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
    vloader = DataLoader(val, batch_size=cfg['batch_size'], num_workers=0)
    model = build_model(cfg['model']).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    history, best = [], float('inf')
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for epoch in range(cfg['epochs']):
        model.train(); loss_sum = n = 0
        t0 = time.perf_counter()
        for x,y,v in loader:
            x,y,v = x.to(args.device),y.to(args.device),v.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = masked_loss(logits,y,v,cfg.get('pos_weight'),cfg.get('dice_weight',1))
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite loss')
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach())*len(x); n += len(x)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        seconds = time.perf_counter()-t0
        report = evaluate(model,vloader,args.device,names_for(cfg['model']),cfg.get('threshold',.5))
        entry = dict(epoch=epoch+1,train_loss=loss_sum/n,train_seconds=seconds,train_images_per_second=n/seconds,validation=report)
        history.append(entry)
        checkpoint = dict(state_dict=model.state_dict(),config=cfg,epoch=epoch+1,
                          evidence='synthetic_smoke' if args.synthetic_smoke else 'real_manifest',
                          manifest_sha256=file_hash(args.manifest))
        torch.save(checkpoint,out/'last.pt')
        if report['loss'] < best:
            best=report['loss']; torch.save(checkpoint,out/'best.pt')
        print(f"epoch={epoch+1} train={loss_sum/n:.5f} val={report['loss']:.5f} train_img_s={n/seconds:.2f}",flush=True)
    write_json(out/'report.json',dict(config=cfg,evidence=checkpoint['evidence'],device=args.device,
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
        manifest_sha256=checkpoint['manifest_sha256'],train_samples=len(train),val_samples=len(val),
        valid_train_pixels=coverage.tolist(),positive_train_pixels=positives.tolist(),
        total_seconds=time.perf_counter()-start,history=history))


if __name__ == '__main__':
    main()
