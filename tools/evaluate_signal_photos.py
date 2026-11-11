"""Evaluate genuine photo ROIs; reports attribute accuracy, never detector recall.

The labels deliberately remain outside the classifier. No geometric template is
created from these photographs. Development and held-out scores stay separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from competition_ai.multi_signal import Candidate, MultiSignalTracker
from competition_ai.signal_shape import ContourTemplateBackend
from competition_ai.signal_association import associate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=Path('docs/multi_signal/photo_labels.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    failures = args.output/'failures'
    failures.mkdir()
    labels = json.loads(args.labels.read_text(encoding='utf-8'))['images']
    sources = {r['id']: r for r in json.loads((args.images/'sources.json').read_text(encoding='utf-8'))}
    backend = ContourTemplateBackend()
    counters, tiles, elapsed = {}, [], []
    with (args.output/'frames.jsonl').open('x', encoding='utf-8') as stream:
        for frame_id, row in enumerate(labels):
            path = args.images/(row['id']+'.jpg')
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Photo hash mismatch: '+str(path))
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Invalid photo: '+str(path))
            counts = counters.setdefault(row['split'], Counter())
            # Still images provide no temporal confirmation; explicitly set
            # one hit for attribute evaluation, rather than duplicating frames.
            tracker = MultiSignalTracker(backend=backend, confirm_updates=1, split_heads=False)
            ns = 1_000_000_000+frame_id*1_000_000_000
            candidates = [Candidate(tuple(t['bbox']), 1., frame_id, ns, 'MANUAL_ROI') for t in row['objects']]
            started = time.perf_counter()
            tracker.update(image, candidates, frame_id=frame_id, timestamp_ns=ns, context=('PHOTO', row['id']))
            result = associate(tracker.snapshot(ns), now_ns=ns)
            elapsed.append((time.perf_counter()-started)*1000)
            result.update(image_id=row['id'], split=row['split'], image_sha256=row['sha256'],
                          input_kind='MANUAL_NATIVE_ROIS_NO_DETECTOR', source_url=sources[row['id']]['page'])
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
            overlay = image.copy()
            for index, truth in enumerate(row['objects']):
                x1, y1, x2, y2 = truth['bbox']
                attr = backend.classify(image[y1:y2, x1:x2])
                counts['objects'] += 1
                for key in ('color', 'shape'):
                    wanted, got = truth[key], getattr(attr, key)
                    if wanted == 'UNKNOWN':
                        counts[key+'_unknown_target_total'] += 1
                        counts[key+'_unknown_target_correct'] += got == 'UNKNOWN'
                        continue
                    counts[key+'_total'] += 1
                    counts[key+'_correct'] += got == wanted
                    counts[key+'_unknown'] += got == 'UNKNOWN'
                    counts[key+'_wrong_known'] += got != wanted and got != 'UNKNOWN'
                    if key == 'shape' and wanted.endswith('_ARROW'):
                        counts['arrow_total'] += 1
                        counts['arrow_correct'] += got == wanted
                        counts['arrow_unknown'] += got == 'UNKNOWN'
                        counts['arrow_wrong_known'] += got != wanted and got != 'UNKNOWN'
                good = attr.color == truth['color'] and attr.shape == truth['shape']
                color = (60, 220, 80) if good else (0, 180, 255)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, max(1, image.shape[1]//700))
                cv2.putText(overlay, str(index), (x1, max(16, y1-4)), 0, max(.4, image.shape[1]/1800), color, 2)
                tile = np.zeros((310, 360, 3), np.uint8)
                crop = image[y1:y2, x1:x2]
                scale = min(340/crop.shape[1], 185/crop.shape[0])
                crop = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))))
                tile[115:115+crop.shape[0], (360-crop.shape[1])//2:(360-crop.shape[1])//2+crop.shape[1]] = crop
                lines = [row['id']+'/'+str(index)+' '+row['split'],
                         'GT: '+truth['color']+' '+truth['shape'],
                         'OUT: '+attr.color+' '+attr.shape, attr.reason]
                for n, line in enumerate(lines):
                    cv2.putText(tile, line, (6, 20+n*24), 0, .45, color if n >= 2 else (240, 240, 240), 1)
                tiles.append(tile)
                if not good:
                    stem = row['id']+'_'+str(index)
                    cv2.imwrite(str(failures/(stem+'.png')), image[y1:y2, x1:x2])
                    (failures/(stem+'.json')).write_text(json.dumps(dict(truth=truth, predicted=asdict(attr),
                        source=sources[row['id']]['page']), ensure_ascii=False, indent=2), encoding='utf-8')
            scale = min(1., 1600/image.shape[1])
            cv2.imwrite(str(args.output/(row['id']+'_annotated.jpg')),
                        cv2.resize(overlay, (round(image.shape[1]*scale), round(image.shape[0]*scale))))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(args.output/'comparison.jpg'), np.vstack([np.hstack(tiles[n:n+4]) for n in range(0, len(tiles), 4)]))
    metrics = {}
    for split, counts in counters.items():
        metrics[split] = dict(counts)
        for key in ('color', 'shape', 'arrow'):
            total = counts[key+'_total']
            metrics[split][key+'_accuracy'] = counts[key+'_correct']/total if total else None
    report = dict(scope='REAL_PHOTO_MANUAL_ROI_ATTRIBUTE_EVALUATION', splits=metrics,
        detector_recall=None, direction_association_accuracy=None,
        exclusions=['NO_DETECTOR_RUN', 'NO_EGO_ROAD_LABELS', 'STILLS_DO_NOT_TEST_TRACKING',
                    'SMALL_NONRANDOM_SAMPLE_NOT_COMPETITION_ACCEPTANCE'],
        attribute_ms_mean=float(np.mean(elapsed)), source_images=len(labels))
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
