"""Restore PR #4's missing-task semantics in mixed detection/segmentation JSON."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .dataset_adapter import read_index, read_json, select_frame, frame_labels, write_json


def audit_one(row):
    row = dict(row)
    if row.get('source') != 'bdd100k' or not row.get('label_json'):
        return row
    frame = select_frame(read_json(row['label_json']), Path(row['image']).name)
    if 'objects' not in frame:
        return row
    labels = frame_labels(frame)
    drivable = any(r.get('category') in ('area/drivable', 'area/alternative') for r in labels)
    lane = any(r.get('category', '').startswith('lane/') for r in labels)
    row['task_available'] = dict(drivable=drivable, lane=lane,
                                 crosswalk='positive-only weak curve')
    if not drivable or not lane:
        path = Path(row['masks'])
        with np.load(path, allow_pickle=False) as data:
            masks = {k: data[k] for k in data.files}
        if not drivable:
            masks['valid'][0] = 0
        if not lane:
            masks['valid'][1:3] = 0
            masks['lane_valid'][:] = 0
        tmp = path.with_name(path.stem + '.validity.tmp.npz')
        np.savez_compressed(tmp, **masks)
        os.replace(tmp, path)
        row['valid_pixels'] = [int(c.sum()) for c in masks['valid']]
        row['positive_pixels'] = [int((a*b).sum()) for a,b in zip(masks['target'], masks['valid'])]
    return row


def audit(manifest, workers=12):
    manifest = Path(manifest)
    marker = manifest.with_name('task-availability-audit.json')
    if marker.exists() and read_json(marker).get('version') == 1:
        return read_json(marker)
    rows = read_index(manifest)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(audit_one, rows))
    tmp = manifest.with_suffix('.audit.tmp.jsonl')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    os.replace(tmp, manifest)
    report = dict(version=1, samples=len(rows),
                  missing_drivable=sum(r.get('task_available', {}).get('drivable') is False for r in rows),
                  missing_lane=sum(r.get('task_available', {}).get('lane') is False for r in rows),
                  policy='Mixed objects JSON without lane/area annotations has zero validity for that task; no invented background labels')
    write_json(marker, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest, args.workers), indent=2), flush=True)


if __name__ == '__main__':
    main()
