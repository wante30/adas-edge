"""Explicit source schemas -> native-size multi-label masks + per-pixel validity.

See reports/phase0.md for source links. No implicit category availability.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from . import CLASSES


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_index(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    if Path(path).suffix.lower() != '.jsonl':
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError('index JSON must contain a list')
        return value
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def poly_points(poly):
    """Flatten modern dict or legacy [x,y,type] Scalabel L/C paths."""
    if isinstance(poly, dict):
        vertices = np.asarray(poly['vertices'], dtype=np.float64)
        types = poly.get('types', 'L' * len(vertices))
    else:
        raw = np.asarray(poly, dtype=object)
        if raw.ndim != 2 or raw.shape[1] not in (2, 3):
            raise ValueError('legacy poly2d must contain [x,y,(type)] vertices')
        vertices = np.asarray(raw[:, :2], dtype=np.float64)
        types = ''.join(str(x[2]) if len(x) > 2 else 'L' for x in poly)
    if len(vertices) < 2 or vertices.shape[1:] != (2,) or len(types) != len(vertices):
        raise ValueError('invalid poly2d vertices/types')
    if not np.isfinite(vertices).all() or set(types) - {'L', 'C'}:
        raise ValueError('unsupported poly2d; expected finite L/C path')
    result = [vertices[0]]
    i = 1
    while i < len(vertices):
        if types[i] == 'L':
            result.append(vertices[i]); i += 1
            continue
        controls = []
        while i < len(vertices) and types[i] == 'C':
            controls.append(vertices[i]); i += 1
        end = vertices[i] if i < len(vertices) else controls[-1]
        if len(controls) >= 3:
            a, b, c, d = result[-1], controls[-3], controls[-2], end
            steps = min(4096, max(8, int(np.linalg.norm(b-a)+np.linalg.norm(c-b)+np.linalg.norm(d-c))))
            for t in np.linspace(0, 1, steps + 1)[1:]:
                result.append((1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*d)
        elif len(controls) == 2:
            a, b, c = result[-1], controls[-2], end
            steps = min(4096, max(8, int(np.linalg.norm(b-a)+np.linalg.norm(c-b))))
            for t in np.linspace(0, 1, steps + 1)[1:]:
                result.append((1-t)**2*a+2*(1-t)*t*b+t*t*c)
        elif controls:
            result.append(controls[0])
        result.append(end)
        if i < len(vertices):
            i += 1
    return np.rint(result).astype(np.int32)


def raster(poly, shape, width=8, force_closed=False):
    layer = np.zeros(shape, np.uint8)
    pts = poly_points(poly)
    if force_closed or (poly.get('closed', False) if isinstance(poly, dict) else False):
        if len(pts) < 3:
            raise ValueError('polygon needs >=3 points')
        cv2.fillPoly(layer, [pts], 1)
    else:
        cv2.polylines(layer, [pts], False, 1, max(1, int(width)), cv2.LINE_8)
    return layer


def polyline_list(polys):
    """Normalize BDD's flat polyline and list-of-polylines serializations."""
    if not polys:
        return []
    first = polys[0]
    if isinstance(first, dict):
        return polys
    if isinstance(first, (list, tuple)) and first and isinstance(first[0], (int, float)):
        return [polys]
    return polys


def frame_labels(frame):
    """Read the modern frame labels or historical objects/frames wrappers."""
    if isinstance(frame, list):
        if len(frame) != 1:
            raise ValueError('lane JSON list must contain exactly one frame')
        frame = frame[0]
    if isinstance(frame, dict) and 'frames' in frame:
        if len(frame['frames']) != 1:
            raise ValueError('lane JSON frames wrapper must contain exactly one frame')
        frame = frame['frames'][0]
    labels = frame.get('labels', frame.get('objects')) if isinstance(frame, dict) else None
    if labels is None:
        raise ValueError('missing lane labels/objects; null means unavailable task')
    return labels


def select_frame(data, image_name):
    """Select exactly one historical multi-frame JSON record by image name."""
    if isinstance(data, list):
        matches = [frame for frame in data if frame.get('name') == image_name]
        if len(matches) != 1:
            raise ValueError(f'expected one BDD frame named {image_name}, got {len(matches)}')
        return matches[0]
    if isinstance(data, dict) and 'frames' in data:
        if data.get('name') == Path(image_name).stem and len(data['frames']) == 1:
            return data['frames'][0]
        matches = [frame for frame in data['frames'] if frame.get('name') == image_name]
        if len(matches) != 1:
            raise ValueError(f'expected one BDD frame named {image_name}, got {len(matches)}')
        return matches[0]
    return data


def bdd_lane_png(out, path):
    """Decode prior category/style PNG as positive-only labels (255 is unknown)."""
    a = np.asarray(Image.open(path))
    if a.shape != out['lane'].shape or a.ndim != 2:
        raise ValueError('expected native-size single-channel BDD lane mask')
    known = (a != 255) & ((a & 0x0f) < 8)
    # Historical renderer uses 255 for the entire background, so only labeled
    # line pixels are valid; never infer negative line examples from that PNG.
    out['lane_valid'][known] = 1
    out['lane'][known] = 1
    style = (a >> 4) & 1
    for value in (0, 1):
        mask = known & (style == value)
        out['target'][1 + value, mask] = 1
        out['valid'][1 + value, mask] = 1


def empty_targets(shape):
    return {k: np.zeros((4, *shape), np.uint8) for k in ('target', 'valid')} | {
        'lane': np.zeros(shape, np.uint8), 'lane_valid': np.zeros(shape, np.uint8)}


def bdd_targets(shape, *, drivable=None, drivable_frame=None, lane_frame=None, lane_png=None, line_width=8):
    out = empty_targets(shape)
    y, v = out['target'], out['valid']
    if drivable is not None:
        a = np.asarray(Image.open(drivable)) if isinstance(drivable, (str, Path)) else np.asarray(drivable)
        if a.shape != shape or not np.isin(a, [0, 1, 2, 255]).all():
            raise ValueError('expected native BDD drivable IDs 0/1/2/255')
        y[0] = (a == 0) | (a == 1)
        v[0] = a != 255
    if drivable_frame is not None:
        if drivable is not None:
            raise ValueError('choose drivable PNG or raw JSON, not both')
        v[0] = int('objects' not in drivable_frame or any(
            label.get('category') in ('area/drivable', 'area/alternative')
            for label in frame_labels(drivable_frame)))
        for label in frame_labels(drivable_frame):
            category = label.get('category', '')
            if category not in ('area/drivable', 'area/alternative'):
                continue
            geometries = polyline_list(label.get('poly2d') or [])
            if not geometries:
                raise ValueError('drivable polygon has no poly2d geometry')
            for poly in geometries:
                mask = raster(poly, shape, force_closed=True)
                y[0] |= mask
    if lane_frame is not None:
        labels = frame_labels(lane_frame)
        available = 'objects' not in lane_frame or any(label.get('category', '').startswith('lane/') for label in labels)
        v[1:3] = int(available)
        out['lane_valid'][:] = int(available)
        ignored = np.zeros(shape, np.uint8)
        for label in labels:
            if not label.get('category', '').startswith('lane/') and label.get('category', '') not in (
                    'crosswalk', 'double other', 'double white', 'double yellow', 'single other', 'single white', 'single yellow'):
                continue
            category = label.get('category', '').removeprefix('lane/')
            if category == 'crosswalk':
                geometries = polyline_list(label.get('poly2d') or [])
                if not geometries:
                    raise ValueError('crosswalk without poly2d geometry')
                for poly in geometries:
                    positive = raster(poly, shape, line_width)
                    y[3] |= positive
                    # BDD crosswalk curves are positive-only weak supervision.
                    v[3, positive > 0] = 1
                continue
            if category in ('road curb', 'background'):
                continue
            if category not in ('double other', 'double white', 'double yellow', 'single other', 'single white', 'single yellow'):
                raise ValueError(f'unknown lane category: {category}')
            attrs = label.get('attributes') or {}
            geometries = polyline_list(label.get('poly2d') or [])
            if not geometries:
                raise ValueError('lane without supported poly2d; convert RLE explicitly first')
            for poly in geometries:
                mask = raster(poly, shape, line_width)
                if attrs.get('ignored') or attrs.get('crowd'):
                    ignored |= mask
                    continue
                out['lane'] |= mask
                style = attrs.get('laneStyle', attrs.get('style'))
                if style in ('solid', 'dashed'):
                    y[1 if style == 'solid' else 2] |= mask
                else:
                    # Unknown styles can train the original lane task, not either style.
                    v[1:3, mask > 0] = 0
        v[1:3, ignored > 0] = 0
        out['lane_valid'][ignored > 0] = 0
    if lane_png is not None:
        if lane_frame is not None:
            raise ValueError('choose raw lane JSON or legacy lane PNG, not both')
        bdd_lane_png(out, lane_png)
    # BDD curves supervise only annotated positive pixels; dense crosswalk pixels stay unknown.
    return out


def drivable_color_ids(path):
    """Decode official BDD visualization colors without guessing unknown pixels."""
    a = np.asarray(Image.open(path).convert('RGB'))
    ids = np.full(a.shape[:2], 255, np.uint8)
    ids[np.all(a == (255, 0, 0), axis=2)] = 0
    ids[np.all(a == (0, 0, 255), axis=2)] = 1
    ids[np.all(a == (0, 0, 0), axis=2)] = 2
    if ids.size == 0 or np.any(ids == 255):
        colors = np.unique(a.reshape(-1, 3), axis=0)
        raise ValueError(f'unrecognized colors in drivable colormap: {colors[:16].tolist()}')
    return ids


def vistas_targets(shape, label_path, config_path, mapping):
    """IDs derived by names in downloaded Vistas config, never guessed by version.

    mapping has only explicitly supervised channels, e.g. crosswalk labels.
    Generic markings do not provide solid/dashed annotations.
    """
    out = empty_targets(shape)
    labels = read_json(config_path)['labels']
    names = {x['name']: i for i, x in enumerate(labels)}
    a = np.asarray(Image.open(label_path))
    if a.shape != shape or a.ndim != 2:
        raise ValueError('expected Vistas semantic class ID PNG (not RGB/instance PNG)')
    known = np.isin(a, [i for i, x in enumerate(labels) if not x.get('evaluate', True) is False and not x['name'].startswith('void--')])
    for channel, positive_names in mapping.items():
        if channel not in CLASSES or not positive_names or set(positive_names) - names.keys():
            raise ValueError(f'invalid Vistas mapping: {channel}: {positive_names}')
        if channel in ('solid_lane', 'dashed_lane'):
            raise ValueError('generic Vistas labels cannot establish lane style; use audited custom masks')
        c = CLASSES.index(channel)
        out['target'][c] = np.isin(a, [names[n] for n in positive_names])
        out['valid'][c] = known
    return out


def validate_manifest(rows, *, check_files=True):
    groups, images, ids, content = {}, {}, set(), {}
    for row in rows:
        split = row['split']
        if split not in ('train', 'val', 'test') or not row['group_id'] or not row['license']:
            raise ValueError('split, sequence group_id and license required')
        # A source prefix cannot be used to hide duplicated groups/frames.
        group = row['group_id']
        image = str(Path(row['image']).resolve()).casefold()
        if group in groups and groups[group] != split:
            raise ValueError(f'video/sequence leakage: {group}')
        if image in images or row['id'] in ids:
            raise ValueError('duplicate image or id')
        groups[group], images[image] = split, split
        ids.add(row['id'])
        if check_files:
            label_key = 'masks' if row.get('masks') else ('drivable_png' if row.get('drivable_png') else 'drivable_color_png')
            for key in ('image', label_key):
                if not Path(row[key]).is_file():
                    raise ValueError(f'missing {key}: {row[key]}')
            digest = file_hash(row['image'])
            if digest in content:
                if content[digest] != split:
                    raise ValueError('duplicate image content across splits (possible leakage)')
            else:
                content[digest] = split
    return rows


@lru_cache(maxsize=4)
def _load_manifest_cached(path, size, mtime_ns):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    return validate_manifest(rows)


def load_manifest(path):
    manifest = Path(path).resolve()
    stat = manifest.stat()
    return copy.deepcopy(_load_manifest_cached(str(manifest), stat.st_size, stat.st_mtime_ns))


def letterbox(image, target=None, valid=None, size=(320, 192)):
    """Shared train/infer/PTQ transform: RGB [0,1], padding ignored by loss."""
    w, h = size
    ih, iw = image.shape[:2]
    scale = min(w/iw, h/ih)
    nw, nh = max(1, round(iw*scale)), max(1, round(ih*scale))
    left, top = (w-nw)//2, (h-nh)//2
    result = np.full((h, w, 3), 114, np.uint8)
    result[top:top+nh, left:left+nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(result.transpose(2, 0, 1).copy()).float()/255
    geometry = dict(native_shape=[ih, iw], resized_shape=[nh, nw], top=top, left=left)
    if target is None:
        return x, geometry
    def resize_masks(a):
        b = np.zeros((len(a), h, w), np.float32)
        for c in range(len(a)):
            b[c, top:top+nh, left:left+nw] = cv2.resize(a[c], (nw, nh), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(b)
    return x, resize_masks(target), resize_masks(valid), geometry


def restore(prob, geometry):
    nh, nw = geometry['resized_shape']
    top, left = geometry['top'], geometry['left']
    ih, iw = geometry['native_shape']
    return np.stack([cv2.resize(c[top:top+nh, left:left+nw], (iw, ih), interpolation=cv2.INTER_LINEAR) for c in prob])


class RoadDataset(Dataset):
    def __init__(self, manifest, split, size=(320, 192), baseline=False, disabled_channels=()):
        self.rows = [r for r in load_manifest(manifest) if r['split'] == split]
        if not self.rows:
            raise ValueError(f'empty {split} split')
        self.size, self.baseline = size, baseline
        self.disabled_channels = set(disabled_channels)
        cache = os.environ.get('BDD_SEG_CACHE')
        self.cache = Path(cache) if cache else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def validate_masks(image, y, v):
        def binary(a):
            return not (a > 1).any() if a.dtype == np.uint8 else np.logical_or(a == 0, a == 1).all()
        if y.shape[1:] != image.shape[:2] or y.shape != v.shape or not binary(y) or not binary(v):
            raise ValueError('native image/mask geometry or binary values mismatch')

    def __getitem__(self, i):
        row = self.rows[i]
        cache_path = None
        if getattr(self, 'cache', None) and row.get('masks'):
            identity = [(str(Path(row[k]).resolve()), Path(row[k]).stat().st_size,
                         Path(row[k]).stat().st_mtime_ns) for k in ('image', 'masks')]
            key = hashlib.sha256(json.dumps(['road-cache-v1', self.size, identity]).encode()).hexdigest()
            cache_path = self.cache / (key + '.npz')
            if cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as m:
                    x = torch.from_numpy(m['rgb'].copy()).float() / 255
                    masks = np.unpackbits(m['bits'], axis=-1, count=self.size[0], bitorder='little')
                y, v = (masks[[0, 8]], masks[[4, 9]]) if self.baseline else (masks[:4], masks[4:8])
                names = ('drivable_area', 'lane') if self.baseline else CLASSES
                for channel in self.disabled_channels:
                    if channel in names:
                        v[names.index(channel)] = 0
                return x, torch.from_numpy(y.astype(np.float32)), torch.from_numpy(v.astype(np.float32))
        image = np.asarray(Image.open(row['image']).convert('RGB'))
        if row.get('masks'):
            with np.load(row['masks'], allow_pickle=False) as m:
                y, v = m['target'], m['valid']
                self.validate_masks(image, y, v)
                if cache_path:
                    rgb, all_y, all_v, _ = letterbox(image, np.concatenate([y, m['lane'][None]]),
                        np.concatenate([v, m['lane_valid'][None]]), self.size)
                    ordered = np.concatenate([all_y[:4], all_v[:4], all_y[4:5], all_v[4:5]]).astype(np.uint8)
                    temporary = cache_path.with_name(cache_path.stem + f'.{os.getpid()}.tmp.npz')
                    np.savez(temporary, rgb=(rgb.numpy()*255).round().astype(np.uint8),
                             bits=np.packbits(ordered, axis=-1, bitorder='little'))
                    os.replace(temporary, cache_path)
                if self.baseline:
                    y, v = np.stack([y[0], m['lane']]), np.stack([v[0], m['lane_valid']])
        else:
            label = row['drivable_png'] if row.get('drivable_png') else drivable_color_ids(row['drivable_color_png'])
            masks = bdd_targets(image.shape[:2], drivable=label)
            y, v = masks['target'], masks['valid']
            if self.baseline:
                y, v = np.stack([y[0], masks['lane']]), np.stack([v[0], masks['lane_valid']])
        if not row.get('masks'):
            self.validate_masks(image, y, v)
        output_names = ('drivable_area', 'lane') if self.baseline else CLASSES
        for channel in self.disabled_channels:
            if channel in output_names:
                v[output_names.index(channel)] = 0
        x, y, v, _ = letterbox(image, y, v, self.size)
        return x, y, v


def _convert_one(task):
    i, row, index_parent, output, line_width, resume = task
    row = dict(row)
    source_path = lambda p: str((index_parent / p).resolve())
    row['image'] = source_path(row['image'])
    mask_path = output / f'{i:08d}.npz'
    if resume and mask_path.exists():
        with np.load(mask_path, allow_pickle=False) as data:
            masks = {key: data[key] for key in data.files}
    else:
        with Image.open(row['image']) as im:
            shape = im.height, im.width
        source = row['source']
        if source == 'bdd100k':
            raw = read_json(source_path(row['label_json'])) if row.get('label_json') else None
            lane = read_json(source_path(row['lane_json'])) if row.get('lane_json') else raw
            if raw is not None:
                raw = select_frame(raw, Path(row['image']).name)
            if lane is not None:
                lane = select_frame(lane, Path(row['image']).name)
            masks = bdd_targets(shape, drivable=source_path(row['drivable_png']) if row.get('drivable_png') else None,
                                drivable_frame=raw, lane_frame=lane,
                                lane_png=source_path(row['lane_png']) if row.get('lane_png') else None,
                                line_width=line_width)
        elif source == 'mapillary_vistas':
            masks = vistas_targets(shape, source_path(row['label_png']), source_path(row['config']), row['mapping'])
        else:
            raise ValueError(f'unsupported source: {source}')
        temporary = mask_path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, **masks)
        os.replace(temporary, mask_path)
    row.update(masks=str(mask_path), image_sha256=file_hash(row['image']),
               valid_pixels=[int(c.sum()) for c in masks['valid']],
               positive_pixels=[int((a*b).sum()) for a,b in zip(masks['target'], masks['valid'])],
               line_width_native_px=line_width)
    return row


def convert_index(index, output, line_width=10, resume=False, workers=0):
    """Index must explicitly enumerate frozen splits and task availability."""
    index = Path(index).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=resume)
    source_marker = output / 'conversion-source.json'
    source_hash = file_hash(index)
    if source_marker.exists():
        if read_json(source_marker).get('index_sha256') != source_hash:
            raise ValueError('resume output belongs to a different source index')
        if read_json(source_marker).get('line_width_native_px') != line_width:
            raise ValueError('resume output uses a different raster line width')
    elif resume and any(output.glob('*.npz')):
        raise ValueError('resume output has masks but no source identity marker')
    write_json(source_marker, dict(index_sha256=source_hash, line_width_native_px=line_width))
    index_rows = read_index(index)
    tasks = [(i, row, index.parent, output, line_width, resume) for i, row in enumerate(index_rows)]
    pending = output / 'converted-rows.jsonl'
    if resume and pending.exists():
        rows = read_index(pending)
    elif workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_convert_one, tasks, chunksize=8))
    else:
        rows = []
        for i, task in enumerate(tasks, 1):
            rows.append(_convert_one(task))
            if i % 2000 == 0:
                print(f'converted={i}/{len(tasks)}', flush=True)
    if not pending.exists():
        pending.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    # Original BDD splits contain a small number of shared video prefixes.
    # Preserve the official val holdout and quarantine train-side frames from
    # those sequences. Raw images/labels/masks and the full source index remain.
    official_val_groups = {r['group_id'] for r in rows
                           if r.get('source') == 'bdd100k' and r.get('original_split') == 'val'}
    excluded = [r for r in rows if r.get('source') == 'bdd100k'
                and r.get('original_split') == 'train' and r['group_id'] in official_val_groups]
    excluded_ids = {r['id'] for r in excluded}
    rows = [r for r in rows if r['id'] not in excluded_ids]
    write_json(output / 'split-audit.json', dict(
        excluded_train_frames=excluded,
        reason='quarantine original train frames whose video prefix occurs in official val',
        split_counts={s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')}))
    validate_manifest(rows)
    manifest = output / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    write_json(output/'provenance.json', dict(index_sha256=file_hash(index), class_names=CLASSES, line_width_native_px=line_width, samples=len(rows)))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--line-width', type=int, default=10)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--workers', type=int, default=0)
    args = p.parse_args()
    if args.line_width <= 0:
        p.error('--line-width must be positive')
    if args.workers < 0:
        p.error('--workers must be non-negative')
    print(convert_index(args.index, args.output, args.line_width, args.resume, args.workers))


if __name__ == '__main__':
    main()
