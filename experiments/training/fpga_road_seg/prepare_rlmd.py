"""Audit public RLMD RGB masks and build an isolated, task-masked mixed manifest."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .dataset_adapter import file_hash, read_index, validate_manifest, write_json

MAPPING = {1: [4, 5, 6, 7, 8], 2: [9, 10], 3: [2]}
COLORS = np.array([[0, 210, 100], [245, 80, 50], [0, 155, 255], [235, 60, 220]])
_lut = None


def convert(task):
    global _lut
    image, label, dest, classes = task
    if _lut is None:
        _lut = np.full(1 << 24, 255, np.uint8)
        for c in classes:
            _lut[(int(c['r']) << 16) | (int(c['g']) << 8) | int(c['b'])] = int(c['id'])
    rgb = np.asarray(Image.open(image).convert('RGB'))
    raw = np.asarray(Image.open(label).convert('RGB'), dtype=np.uint32)
    if raw.shape != rgb.shape:
        raise ValueError(f'geometry mismatch: {image}')
    ids = _lut[(raw[..., 0] << 16) | (raw[..., 1] << 8) | raw[..., 2]]
    target = np.zeros((4, *ids.shape), np.uint8)
    valid = np.zeros_like(target)
    for channel, positive_ids in MAPPING.items():
        target[channel] = np.isin(ids, positive_ids)
        valid[channel] = ids != 255
    lane = np.isin(ids, MAPPING[1] + MAPPING[2]).astype(np.uint8)
    if not dest.exists():
        np.savez_compressed(dest, target=target, valid=valid, lane=lane, lane_valid=(ids != 255).astype(np.uint8))
    gray = cv2.cvtColor(cv2.resize(rgb, (9, 8)), cv2.COLOR_RGB2GRAY)
    dhash = int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), 'big')
    return dict(id='rlmd-' + image.stem, source='rlmd', image=str(image), masks=str(dest),
                original_split=image.parent.name, image_sha256=file_hash(image), label_sha256=file_hash(label),
                valid_pixels=valid.sum((1, 2)).tolist(), positive_pixels=(target * valid).sum((1, 2)).tolist(),
                unknown_pixels=int((ids == 255).sum()), dhash=str(dhash),
                license='Author public research release; no explicit official license; mirror MIT claim unverified; no redistribution')


def grouped_splits(rows):
    # Filenames do not identify recording sessions. Consecutive numeric runs are
    # conservatively kept together; original random frame split is retained as metadata.
    numeric = sorted((r for r in rows if Path(r['image']).stem.isdigit()), key=lambda r: int(Path(r['image']).stem))
    previous, run = None, 0
    for row in numeric:
        number = int(Path(row['image']).stem)
        if previous is not None and number - previous > 500:
            run += 1
        row['group_id'] = f'rlmd-numeric-run-{run}'
        previous = number
    for row in rows:
        if 'group_id' not in row:
            row['group_id'] = 'rlmd-' + re.sub(r'_\d+$', '', Path(row['image']).stem)
    groups = sorted({r['group_id'] for r in rows}, key=lambda g: hashlib.sha256(('rlmd-stage2-15:' + g).encode()).hexdigest())
    selected, n = set(), 0
    for g in groups:
        count = sum(r['group_id'] == g for r in rows)
        if n == 0 or abs(n + count - len(rows)*.2) < abs(n - len(rows)*.2):
            selected.add(g); n += count
    for r in rows:
        r['split'] = 'val' if r['group_id'] in selected else 'train'
    # Exact duplicate content must never cross the new split. Quarantine also
    # very close dHashes across splits; all exclusions are retained in the audit.
    val = [r for r in rows if r['split'] == 'val']
    quarantined = []
    for r in rows:
        if r['split'] != 'train':
            continue
        matches = [v['id'] for v in val if r['image_sha256'] == v['image_sha256'] or
                   (int(r['dhash']) ^ int(v['dhash'])).bit_count() <= 2]
        if matches:
            quarantined.append(dict(id=r['id'], matches=matches, reason='cross-split identical image or dHash distance <=2'))
    excluded = {r['id'] for r in quarantined}
    return [r for r in rows if r['id'] not in excluded], quarantined


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--bdd-manifest', required=True)
    p.add_argument('--output', required=True); p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    root, out = Path(args.root).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True); (out/'masks').mkdir(exist_ok=True)
    classes = list(csv.DictReader((root/'official_classes.csv').open(encoding='utf-8-sig')))
    native = root/'extracted/RLMD_1080p'
    mirror_classes = list(csv.reader((native/'rlmd.csv').open(encoding='utf-8-sig')))
    assert [[c['id'], c['name'], c['r'], c['g'], c['b']] for c in classes] == mirror_classes, 'official/mirror class mismatch'
    tasks = []
    for split in ('train', 'val'):
        images = {p.stem: p for p in (native/'images'/split).glob('*.jpg')}
        masks = {p.stem: p for p in (native/'labels'/split).glob('*.png')}
        if images.keys() != masks.keys():
            raise ValueError(f'unpaired {split}: images-only={images.keys()-masks.keys()}, labels-only={masks.keys()-images.keys()}')
        tasks.extend((image, masks[stem], out/'masks'/f'{stem}.npz', classes) for stem, image in sorted(images.items()))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(convert, tasks, chunksize=4), 1):
            rows.append(row)
            if i % 200 == 0:
                print(f'converted={i}/{len(tasks)}', flush=True)
    original_counts = Counter(r['original_split'] for r in rows)
    rows, excluded = grouped_splits(rows)
    validate_manifest(rows)
    for split in ('train', 'val'):
        positives = np.array([r['positive_pixels'] for r in rows if r['split'] == split]).sum(0)
        if (positives[1:] <= 0).any():
            raise ValueError(f'{split} lacks requested dense class: {positives}')
    def save(name, data):
        (out/name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in data), encoding='utf-8')
    save('rlmd_manifest.jsonl', rows)
    bdd = read_index(args.bdd_manifest)
    for row in bdd:
        row['disabled_channels'] = sorted(set(row.get('disabled_channels', [])) | {'crosswalk'})
    save('mixed_manifest.jsonl', bdd + rows)
    audit = dict(original_counts=dict(original_counts), samples=len(rows), split_counts=dict(Counter(r['split'] for r in rows)),
                 mapping=MAPPING, drivable_valid_pixels=0, unknown_pixels=sum(r['unknown_pixels'] for r in rows),
                 split_rule='Numeric contiguous runs separated by >500 frame index; named day sequences grouped. Deterministic 20% group holdout; unknown true recording IDs, no claim of complete scene independence.',
                 excluded=excluded, official_classes_sha256=file_hash(root/'official_classes.csv'),
                 validation_groups=sorted({r['group_id'] for r in rows if r['split']=='val'}),
                 counts={s: dict(positive_pixels=np.array([r['positive_pixels'] for r in rows if r['split']==s]).sum(0).tolist(),
                                 valid_pixels=np.array([r['valid_pixels'] for r in rows if r['split']==s]).sum(0).tolist()) for s in ('train','val')})
    write_json(out/'audit.json', audit)
    selected = [r for r in rows if r['split']=='val' and r['positive_pixels'][3] > 500][:12]
    canvas = Image.new('RGB', (1440, 4*300), 'white'); draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(selected):
        rgb = np.asarray(Image.open(row['image']).convert('RGB').resize((480,270))).astype(float)
        with np.load(row['masks']) as m:
            for c in range(1,4):
                active = cv2.resize(m['target'][c], (480,270), interpolation=cv2.INTER_NEAREST) > 0
                rgb[active] = rgb[active]*.5 + COLORS[c]*.5
        x,y = i%3*480, i//3*300
        draw.text((x+5,y+5), row['id']+' dense GT', fill='black')
        canvas.paste(Image.fromarray(rgb.astype(np.uint8)), (x,y+30))
    canvas.save(out/'dense_labels_check.jpg')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
