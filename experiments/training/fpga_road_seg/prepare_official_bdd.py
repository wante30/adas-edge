"""Extract original BDD splits and build a sequence-safe mask-rendering index."""
import argparse
import json
import random
import zipfile
from pathlib import Path


LICENSE = 'BDD100K original data license: non-profit educational/research use; Internet Archive public mirror item bdd100k'


def extract_prefix(archive, prefix, suffix, root):
    root = Path(root)
    selected = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename.replace('\\', '/')
            if not name.startswith(prefix) or not name.lower().endswith(suffix):
                continue
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != info.file_size:
                    raise IOError(f'refusing to overwrite mismatched extracted file: {target}')
            else:
                with zf.open(info) as src, open(target, 'wb') as dst:
                    while block := src.read(8 * 1024 * 1024):
                        dst.write(block)
            selected.append(target.resolve())
    return selected


def build_index(root, images_zip, labels_zip, output, val_fraction=0.1, seed=15):
    root, output = Path(root).resolve(), Path(output).resolve()
    raw = root / 'extracted'
    images, labels = {}, {}
    for split in ('train', 'val', 'test'):
        images[split] = extract_prefix(images_zip, f'bdd100k/images/100k/{split}/', '.jpg', raw)
        labels[split] = extract_prefix(labels_zip, f'bdd100k/labels/100k/{split}/', '.json', raw)
        print(f'{split}: images={len(images[split])} json={len(labels[split])}', flush=True)

    label_maps = {s: {p.stem: p for p in labels[s]} for s in ('train', 'val')}
    all_rows = []
    train_groups = sorted({p.stem.split('-', 1)[0] for p in images['train']})
    if not 0 < val_fraction < 0.5:
        raise ValueError('val_fraction must be between 0 and 0.5')
    random.Random(seed).shuffle(train_groups)
    holdout_groups = set(train_groups[:max(1, round(len(train_groups) * val_fraction))])
    missing = []
    for source_split in ('train', 'val'):
        for image in sorted(images[source_split]):
            label = label_maps[source_split].get(image.stem)
            if label is None:
                missing.append((source_split, image.name))
                continue
            group = image.stem.split('-', 1)[0]
            split = ('val' if group in holdout_groups else 'train') if source_split == 'train' else 'test'
            all_rows.append(dict(id=image.stem, source='bdd100k', image=str(image), label_json=str(label),
                                 split=split, group_id=group, license=LICENSE, original_split=source_split,
                                 crosswalk_supervision='weak_positive_curve_only'))
    orphan_labels = {s: sorted(set(label_maps[s]) - {p.stem for p in images[s]}) for s in ('train', 'val')}
    if missing or any(orphan_labels.values()):
        raise ValueError(f'image/JSON join incomplete; missing_labels={missing[:10]} total={len(missing)} orphan_labels={orphan_labels}')
    if len(all_rows) != sum(len(images[s]) for s in ('train', 'val')):
        raise AssertionError('labeled split count mismatch')

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in all_rows), encoding='utf-8')
    unlabeled_test = [dict(id=p.stem, image=str(p), source='bdd100k', source_split='test',
                           lane_supervision='unavailable', drivable_supervision='unavailable')
                      for p in sorted(images['test'])]
    output.with_name('unlabeled_test.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in unlabeled_test), encoding='utf-8')
    summary = dict(samples=len(all_rows), train_images=len(images['train']), val_images=len(images['val']),
                   unlabeled_test_images=len(images['test']), label_json_train=len(labels['train']),
                   label_json_val=len(labels['val']), train_rows=sum(r['split'] == 'train' for r in all_rows),
                   internal_val_rows=sum(r['split'] == 'val' for r in all_rows),
                   heldout_test_rows=sum(r['split'] == 'test' for r in all_rows),
                   holdout_video_groups=len(holdout_groups), seed=seed, val_fraction=val_fraction,
                   crosswalk='weak positive-only curve supervision; not dense zebra-crossing ground truth',
                   lane_style='solid/dashed decoded from original BDD poly2d laneStyle/style attributes')
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--images-zip', required=True)
    parser.add_argument('--labels-zip', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=15)
    args = parser.parse_args()
    result = build_index(args.root, args.images_zip, args.labels_zip, args.output, args.val_fraction, args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
