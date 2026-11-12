"""Train real manifests, or explicitly labelled synthetic smoke fixtures."""
import argparse
from collections import Counter
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

from . import CLASSES
from .dataset_adapter import RoadDataset, read_json, write_json, file_hash, drivable_color_ids
from .engine import evaluate, masked_loss
from .models import build_model


def names_for(model):
    return ('drivable_area', 'lane') if model in ('twin-small', 'twin-nano') else CLASSES


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint['config']['model']).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model, checkpoint


class AugmentedRoadDataset(RoadDataset):
    """Image-only photometric changes and shared horizontal geometry."""
    def __getitem__(self, i):
        x, y, v = super().__getitem__(i)
        if random.random() < .5:
            x, y, v = x.flip(-1), y.flip(-1), v.flip(-1)
        gain, bias = random.uniform(.85, 1.15), random.uniform(-.04, .04)
        return (x * gain + bias).clamp(0, 1), y, v


def selection_score(domains, baseline, tolerance):
    """Select only improvements which respect each original BDD task."""
    b = domains['bdd100k']['classes']
    keys = [('drivable_area', 'iou'), ('solid_lane', 'f1'), ('dashed_lane', 'f1')]
    values = [b[c][m] for c, m in keys]
    old = baseline['bdd100k']['classes']
    eligible = all(value is not None and value >= old[c][m] - tolerance
                   for value, (c, m) in zip(values, keys))
    crosswalk = domains['rlmd']['classes']['crosswalk']['iou']
    score = .4 * (crosswalk or 0) + .2 * sum(v or 0 for v in values)
    return score, eligible


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--synthetic-smoke', action='store_true', help='Label results as synthetic; never a real-data metric')
    p.add_argument('--resume', help='resume from a training checkpoint')
    p.add_argument('--init', help='transfer model weights only; fresh optimizer and manifest, distinct from resume')
    p.add_argument('--teacher', help='optional trained two-head teacher; drivable distillation only on genuinely supervised pixels')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--amp', action='store_true', help='use CUDA mixed precision')
    p.add_argument('--train-limit', type=int, default=0, help='deterministic stage-one subset; zero uses all train samples')
    p.add_argument('--val-limit', type=int, default=0, help='stage-one validation subset; zero uses full internal val')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--exclude-channel', action='append', choices=CLASSES, default=[],
                   help='disable a task loss when labels are weak or unavailable; repeat as needed')
    args = p.parse_args()
    if args.init and args.resume:
        p.error('--init and --resume are mutually exclusive')
    cfg = read_json(args.config)
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    if cfg['epochs'] < 1 or cfg['batch_size'] < 1:
        p.error('epochs and batch_size must be positive')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(4)
    out = Path(args.output)
    if out.exists() and not args.resume and any(out.iterdir()):
        p.error('nonempty output requires --resume or a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    size = cfg['width'], cfg['height']
    baseline = cfg['model'] in ('twin-small', 'twin-nano')
    train_class = AugmentedRoadDataset if cfg.get('augment') else RoadDataset
    train = train_class(args.manifest, 'train', size, baseline, args.exclude_channel)
    val = RoadDataset(args.manifest, 'val', size, baseline, args.exclude_channel)
    if args.train_limit:
        if args.train_limit < 1:
            p.error('--train-limit must be positive or zero')
        train.rows = sorted(train.rows, key=lambda row: row['id'])[:args.train_limit]
    if args.val_limit:
        val.rows = sorted(val.rows, key=lambda row: row['id'])[:args.val_limit]
    if any(r['source'] == 'synthetic' for r in train.rows+val.rows) and not args.synthetic_smoke:
        p.error('synthetic fixture requires --synthetic-smoke')
    channels = len(names_for(cfg['model']))
    coverage = np.zeros(channels, np.int64)
    positives = coverage.copy()
    for row in train.rows:
        if baseline and 'valid_pixels' in row and 'positive_pixels' in row:
            coverage[0] += row['valid_pixels'][0]
            positives[0] += row['positive_pixels'][0]
            with np.load(row['masks'], allow_pickle=False) as masks:
                lane, known = masks['lane'], masks['lane_valid']
                coverage[1] += int(known.sum())
                positives[1] += int((lane * known).sum())
            continue
        if not baseline and 'valid_pixels' in row and 'positive_pixels' in row:
            available = np.array(row['valid_pixels'], dtype=np.int64)
            positive = np.array(row['positive_pixels'], dtype=np.int64)
            for channel in set(args.exclude_channel) | set(row.get('disabled_channels', ())):
                available[CLASSES.index(channel)] = 0
                positive[CLASSES.index(channel)] = 0
            coverage += available
            positives += positive
            continue
        if row.get('masks'):
            with np.load(row['masks'], allow_pickle=False) as masks:
                if baseline:
                    y = np.stack([masks['target'][0], masks['lane']])
                    v = np.stack([masks['valid'][0], masks['lane_valid']])
                else:
                    y, v = masks['target'], masks['valid']
                output_names = names_for(cfg['model'])
                for channel in set(args.exclude_channel) | set(row.get('disabled_channels', ())):
                    if channel in output_names:
                        v[output_names.index(channel)] = 0
                coverage += v.sum((1,2)).astype(np.int64)
                positives += (y*v).sum((1,2)).astype(np.int64)
        else:
            raw = np.asarray(Image.open(row['drivable_png'])) if row.get('drivable_png') else drivable_color_ids(row['drivable_color_png'])
            known, positive = raw != 255, (raw == 0) | (raw == 1)
            coverage[0] += int(known.sum()); positives[0] += int((positive & known).sum())
    # Refuse a four-class claim with absent channels/positive labels.
    if not args.synthetic_smoke and not np.any((coverage > 0) & (positives > 0)):
        p.error(f'no positively supervised training channel: coverage={coverage}, positives={positives}')
    loader_args = dict(batch_size=cfg['batch_size'], num_workers=max(0, args.workers),
                       pin_memory=args.device.startswith('cuda'))
    if args.workers > 0:
        loader_args.update(persistent_workers=True, prefetch_factor=2)
    sampler = None
    if cfg.get('source_probabilities'):
        counts = Counter(r['source'] for r in train.rows)
        probabilities = cfg['source_probabilities']
        if counts.keys() != probabilities.keys() or abs(sum(probabilities.values()) - 1) > 1e-6:
            p.error('source probabilities must cover each training source and sum to one')
        sampler = WeightedRandomSampler([probabilities[r['source']] / counts[r['source']] for r in train.rows],
                                       cfg.get('epoch_samples', len(train)), replacement=True)
    loader = DataLoader(train, shuffle=sampler is None, sampler=sampler, **loader_args)
    domain_loaders = {}
    if cfg.get('domain_validation'):
        for source in sorted({r['source'] for r in val.rows}):
            subset = Subset(val, [i for i, row in enumerate(val.rows) if row['source'] == source])
            domain_loaders[source] = DataLoader(subset, shuffle=False, **{**loader_args, 'num_workers': min(2, args.workers)})
        vloader = None
    else:
        vloader = DataLoader(val, shuffle=False, **loader_args)
    model = build_model(cfg['model']).to(args.device)
    teacher = None
    teacher_identity = None
    if args.teacher:
        teacher, teacher_ck = load_checkpoint(args.teacher,args.device)
        if names_for(teacher_ck['config']['model']) != ('drivable_area','lane'):
            p.error('drivable teacher must be the existing two-head reference model')
        teacher.eval().requires_grad_(False)
        teacher_identity = dict(path=str(Path(args.teacher).resolve()),sha256=file_hash(args.teacher))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    amp_enabled = bool(args.amp and args.device.startswith('cuda'))
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    history, best, start_epoch = [], float('inf'), 0
    baseline_domains, best_score, stale = None, -float('inf'), 0
    initialization = None
    if args.init:
        initial = torch.load(args.init, map_location=args.device, weights_only=False)
        if initial['config']['model'] != cfg['model']:
            p.error('initial checkpoint model mismatch')
        model.load_state_dict(initial['state_dict'], strict=True)
        initialization = dict(path=str(Path(args.init).resolve()), sha256=file_hash(args.init), epoch=initial['epoch'])
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        if checkpoint.get('manifest_sha256') != file_hash(args.manifest) or checkpoint.get('config', {}).get('model') != cfg['model']:
            p.error('resume checkpoint manifest/model mismatch')
        if checkpoint.get('teacher') != teacher_identity:
            p.error('resume teacher mismatch')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        if checkpoint.get('optimizer_state'):
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if checkpoint.get('scaler_state'):
            scaler.load_state_dict(checkpoint['scaler_state'])
        history = checkpoint.get('history', [])
        best = checkpoint.get('best_validation_loss', best)
        if checkpoint.get('val_samples') != len(val) or checkpoint.get('excluded_channels', []) != args.exclude_channel:
            best = float('inf')
        start_epoch = checkpoint['epoch']
        baseline_domains = checkpoint.get('baseline_domains')
        best_score, stale = checkpoint.get('best_score', best_score), checkpoint.get('stale_epochs', 0)
        initialization = checkpoint.get('initialization')
        if checkpoint.get('random_state'):
            random.setstate(checkpoint['random_state'])
            np.random.set_state(checkpoint['numpy_state'])
            torch.set_rng_state(checkpoint['torch_rng'].cpu())
            if checkpoint.get('cuda_rng') is not None and args.device.startswith('cuda'):
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    if domain_loaders and baseline_domains is None:
        baseline_domains = {source: evaluate(model, dl, args.device, names_for(cfg['model']), cfg.get('threshold', .5))
                            for source, dl in domain_loaders.items()}
        best_score, _ = selection_score(baseline_domains, baseline_domains, cfg.get('bdd_regression_tolerance', .025))
        write_json(out/'before_finetune.json', dict(initialization=initialization, domains=baseline_domains))
        print(f'initial_domains={baseline_domains}', flush=True)
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    if start_epoch >= cfg['epochs']:
        print(f'already_complete epoch={start_epoch}', flush=True)
        return
    for epoch in range(start_epoch, cfg['epochs']):
        model.train(); loss_sum = n = 0
        t0 = time.perf_counter()
        for x,y,v in loader:
            x,y,v = x.to(args.device),y.to(args.device),v.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                logits = model(x)
                loss = masked_loss(logits,y,v,cfg.get('pos_weight'),cfg.get('dice_weight',1))
                if teacher is not None:
                    with torch.no_grad():
                        soft = (teacher(x)[:,0:1]/2).sigmoid()
                    known = v[:,0:1]
                    distillation = torch.nn.functional.binary_cross_entropy_with_logits(logits[:,0:1]/2,soft,reduction='none')
                    # RLMD drivable remains absent: valid=0 also masks KD.
                    loss = loss + cfg.get('distillation_weight',.2)*4*(distillation*known).sum()/known.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite loss')
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.detach())*len(x); n += len(x)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        seconds = time.perf_counter()-t0
        domains = {source: evaluate(model, dl, args.device, names_for(cfg['model']), cfg.get('threshold', .5))
                   for source, dl in domain_loaders.items()}
        report = (dict(loss=sum(d['loss'] for d in domains.values())/len(domains),
                       samples=sum(d['samples'] for d in domains.values()), domains=domains)
                  if domains else evaluate(model,vloader,args.device,names_for(cfg['model']),cfg.get('threshold',.5)))
        entry = dict(epoch=epoch+1,train_loss=loss_sum/n,train_seconds=seconds,train_images_per_second=n/seconds,
                     train_samples=n, val_samples=len(val), validation=report)
        history.append(entry)
        improved = report['loss'] < best
        if domains:
            score, eligible = selection_score(domains, baseline_domains, cfg.get('bdd_regression_tolerance', .025))
            improved = eligible and score > best_score + cfg.get('min_delta', .001)
            if improved:
                best_score = score
            stale = 0 if improved else stale + 1
            entry.update(selection_score=score, bdd_guard_passed=eligible, learning_rate=optimizer.param_groups[0]['lr'])
            # Lower LR on a plateau; no architectural change during this run.
            if stale and stale % 2 == 0:
                for group in optimizer.param_groups:
                    group['lr'] = max(cfg['lr'] * .0625, group['lr'] * .5)
        best = min(best, report['loss'])
        checkpoint = dict(state_dict=model.state_dict(),config=cfg,epoch=epoch+1,
                          evidence='synthetic_smoke' if args.synthetic_smoke else 'real_manifest',
                          manifest_sha256=file_hash(args.manifest),optimizer_state=optimizer.state_dict(),
                          scaler_state=scaler.state_dict(),history=history,best_validation_loss=best,
                          amp=amp_enabled, excluded_channels=args.exclude_channel, val_samples=len(val), train_samples=len(train),
                          initialization=initialization, baseline_domains=baseline_domains,
                          teacher=teacher_identity,
                          best_score=best_score if domains else None, stale_epochs=stale,
                          random_state=random.getstate(), numpy_state=np.random.get_state(), torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else None)
        torch.save(checkpoint,out/'last.tmp.pt'); os.replace(out/'last.tmp.pt',out/'last.pt')
        if (epoch+1) % 5 == 0:
            torch.save(checkpoint,out/f'epoch-{epoch+1:03d}.pt')
        if improved:
            torch.save(checkpoint,out/'best.tmp.pt'); os.replace(out/'best.tmp.pt',out/'best.pt')
        print(f"epoch={epoch+1} train={loss_sum/n:.5f} val={report['loss']:.5f} train_img_s={n/seconds:.2f}",flush=True)
        if domains:
            print(f'domains={domains} score={score:.6f} bdd_guard={eligible} stale={stale}', flush=True)
        write_json(out/'report.tmp.json',dict(config=cfg,evidence=checkpoint['evidence'],device=args.device,
            initialization=initialization,teacher=teacher_identity,
            torch_version=torch.__version__,gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
            amp=amp_enabled,workers=args.workers,stage_train_limit=args.train_limit or None,
            excluded_channels=args.exclude_channel, cache=os.environ.get('BDD_SEG_CACHE'),
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
            manifest_sha256=checkpoint['manifest_sha256'],train_samples=len(train),val_samples=len(val),
            valid_train_pixels=coverage.tolist(),positive_train_pixels=positives.tolist(),
            total_seconds=time.perf_counter()-start,history=history))
        os.replace(out/'report.tmp.json',out/'report.json')
        if domains and epoch + 1 >= cfg.get('min_epochs', 10) and stale >= cfg.get('patience', 4):
            print(f'early_stop epoch={epoch+1} stale={stale}', flush=True)
            break
    if domains and not (out/'best.pt').exists():
        write_json(out/'no_eligible_checkpoint.json', dict(reason='No mixed checkpoint improved constrained internal validation; retain stage-one model',
                                                        initialization=initialization))


if __name__ == '__main__':
    main()
