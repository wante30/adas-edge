"""Fresh deployed-lineage ONNX inference + independent manual-ROI diagnostics.

No board, Qt, PCIe, cache from previous PC audits, or control-system imports.
Metrics use only explicitly reviewed frames; video overlays cover every frame.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.offline_signal_models import OfflineDetector, sha256
from competition_ai.multi_signal import Candidate, MultiSignalTracker, iou
from competition_ai.signal_shape import ContourTemplateBackend
from competition_ai.signal_association import associate


def load_baseline(package):
    spec = importlib.util.spec_from_file_location('frozen_signal_ai', Path(package)/'__init__.py',
                                                submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from frozen_signal_ai.signal_shape import ContourTemplateBackend as FrozenBackend
    return FrozenBackend()


def export_baseline(destination, ref):
    destination.mkdir()
    for name in ('__init__.py', 'traffic_light.py', 'multi_signal.py', 'signal_shape.py'):
        (destination/name).write_bytes(subprocess.check_output(
            ['git', 'show', ref+':competition_ai/'+name], cwd=ROOT))
    return destination


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def crop(image, box):
    x1, y1 = np.floor(box[:2]).astype(int)
    x2, y2 = np.ceil(box[2:]).astype(int)
    return image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]


def score_attr(counts, prefix, truth, attr):
    for key in ('color', 'shape'):
        wanted, got = truth[key], attr[key]
        if wanted == 'UNKNOWN':
            continue
        counts[prefix+key+'_total'] += 1
        counts[prefix+key+'_correct'] += got == wanted
        counts[prefix+key+'_unknown'] += got == 'UNKNOWN'
        counts[prefix+key+'_wrong_known'] += got not in (wanted, 'UNKNOWN')
        if key == 'shape' and wanted.endswith('_ARROW'):
            counts[prefix+'arrow_total'] += 1
            counts[prefix+'arrow_correct'] += got == wanted
            counts[prefix+'arrow_unknown'] += got == 'UNKNOWN'
            counts[prefix+'arrow_wrong_known'] += got not in (wanted, 'UNKNOWN')


def matches(rows, truths, threshold=.25):
    """One-to-one strict head IoU .5. No silent bulb/head metric substitution."""
    edges = sorted([(iou(r['bbox'], t['bbox_q4']), ri, ti)
                    for ri, r in enumerate(rows) if r['confidence'] >= threshold
                    for ti, t in enumerate(truths)], reverse=True)
    used_r, used_t, found = set(), set(), {}
    for overlap, ri, ti in edges:
        if overlap >= .5 and ri not in used_r and ti not in used_t:
            found[ti] = ri
            used_r.add(ri)
            used_t.add(ti)
    return found


def text(canvas, line, xy, size=.45, color=(235, 235, 235)):
    cv2.putText(canvas, line, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'docs/multi_signal/offline_eval_manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=['bdd', 'p2', 'specialist'])
    parser.add_argument('--clips', nargs='*')
    parser.add_argument('--stride', type=int, default=1, help='Fresh inference stride; GT frames always run')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    model_root = Path(manifest['model_root'])
    specs = dict(bdd=('rknn_b05o/bdd_yolov8n_512x288_raw_heads.onnx', False, 8),
                 p2=('rknn_b05o/b05n_p2_multiclass_512x288_raw_heads_20261026.onnx', False, 8),
                 specialist=('rknn_b05o/b05l_v8n_p2_512x192_raw_heads_20261026.onnx', False, 0),
                 mtl=('b05n_mtl_rknn_b2a_seg256_20261031/b05n_p2_mtl_b2a_seg256.onnx', False, 8))
    models = {k: OfflineDetector(model_root/specs[k][0], logits=specs[k][1], traffic_light_id=specs[k][2]) for k in args.models}
    dump(args.output/'models.json', {k: m.evidence for k, m in models.items()})
    baseline = load_baseline(export_baseline(args.output/'frozen_baseline', manifest['baseline_ref']))
    improved = ContourTemplateBackend()
    reports = {}
    for clip in manifest['videos']:
        if args.clips and clip['id'] not in args.clips:
            continue
        output = args.output/clip['id']; output.mkdir()
        failures = output/'failures'; failures.mkdir()
        video_hash = sha256(clip['path'])
        if clip.get('sha256') and video_hash != clip['sha256']:
            raise ValueError('Video hash mismatch')
        cap = cv2.VideoCapture(clip['path'])
        if not cap.isOpened():
            raise ValueError('Cannot open '+clip['path'])
        fps, total = cap.get(5), int(cap.get(7))
        labels = {r['frame_id']: r for r in clip['labels']}
        counts, timing, diagnostic_rows, tiles = Counter(), [], [], []
        trackers = {(model, resolution): MultiSignalTracker(backend=improved)
                    for model in models for resolution in ('q4', 'original')}
        results, raw, age, fid = {}, {}, 0, 0
        # Detector panels and GT native/Q4 inset panels in one complete video.
        writer = cv2.VideoWriter(str(output/'comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1920, 880))
        if not writer.isOpened():
            raise RuntimeError('Video writer unavailable')
        with (output/'frames.jsonl').open('x', encoding='utf-8') as stream:
            while True:
                ok, original = cap.read()
                if not ok:
                    break
                q4 = cv2.resize(original, (640, 360), interpolation=cv2.INTER_AREA)
                sx, sy = original.shape[1]/640, original.shape[0]/360
                ns = 1_000_000_000+round(fid/fps*1e9)
                inferred = fid % args.stride == 0 or fid in labels
                if inferred:
                    raw = {}
                    for model, detector in models.items():
                        inp = q4[:216] if model == 'specialist' else q4
                        raw[model], elapsed = detector.detect(inp)
                        timing.append(dict(model=model, frame_id=fid, ms=elapsed))
                    age = fid
                attribute_update = inferred and (fid % max(1, round(fps/2)) == 0 or fid in labels)
                if attribute_update:
                    for (model, resolution), tracker in trackers.items():
                        image = q4 if resolution == 'q4' else original
                        scale = (1, 1) if resolution == 'q4' else (sx, sy)
                        candidates = [Candidate(tuple(v*scale[n%2] for n, v in enumerate(r['bbox'])),
                                      r['confidence'], fid, ns, model) for r in raw[model] if r['confidence'] >= .25]
                        tracker.update(image, candidates, frame_id=fid, timestamp_ns=ns,
                                       context=('OFFLINE', clip['id'], model, resolution))
                results = {model: {resolution: associate(trackers[model, resolution].snapshot(ns), now_ns=ns)
                                     for resolution in ('q4', 'original')} for model in models}
                for by_resolution in results.values():
                    for resolution, result in by_resolution.items():
                        result['input_resolution'] = [640,360] if resolution=='q4' else [original.shape[1],original.shape[0]]
                        result['offline_coordinate_space'] = 'q4_native' if resolution=='q4' else 'source_video_native_reference'
                        if resolution=='original':
                            for signal in result['signals']:
                                signal['bbox_space']='source_video_native_reference'
                evaluation = []
                if fid in labels:
                    truths = [dict(t, bbox_q4=[v/(sx if n%2 == 0 else sy) for n, v in enumerate(t['bbox_native'])])
                              for t in labels[fid]['objects']]
                    counts['reviewed_frames'] += 1
                    counts['gt_objects'] += len(truths)
                    matchsets = {m: matches(rows, truths) for m, rows in raw.items()}
                    for model, found in matchsets.items():
                        for threshold in (.05, .15, .25):
                            counts[model+f'_head_iou50_recall_at{threshold}'] += len(matches(raw[model], truths, threshold))
                    for index, truth in enumerate(truths):
                        entry = dict(frame_id=fid, object_id=truth['id'], truth=truth, manual_roi={}, detections={})
                        native_roi = truth.get('attribute_roi_native', truth['bbox_native'])
                        q4_roi = [v/(sx if n%2 == 0 else sy) for n,v in enumerate(native_roi)]
                        for resolution, image, box in [('q4', q4, q4_roi), ('original', original, native_roi)]:
                            roi = crop(image, box)
                            entry['manual_roi'][resolution] = {}
                            for version, backend in [('baseline', baseline), ('improved', improved)]:
                                try:
                                    attr = asdict(backend.classify(roi))
                                except cv2.error as exc:
                                    if version != 'baseline': raise
                                    attr = dict(color='UNKNOWN', shape='UNKNOWN', reason='BASELINE_OPENCV_ERROR', error=str(exc))
                                entry['manual_roi'][resolution][version] = attr
                                score_attr(counts, f'manual_{resolution}_{version}_', truth, attr)
                            a = entry['manual_roi'][resolution]['improved']
                            tile = np.zeros((240, 440, 3), np.uint8)
                            for n, line in enumerate([f"f{fid} #{truth['id']} {resolution} {roi.shape[1]}x{roi.shape[0]}",
                                    'GT '+truth['color']+' '+truth['shape'],
                                    'OUT '+a['color']+' '+a['shape'], a['reason']]):
                                text(tile, line, (4, 20+22*n), .42)
                            scale = min(420/max(1, roi.shape[1]), 130/max(1, roi.shape[0]))
                            resized = cv2.resize(roi, (max(1, round(roi.shape[1]*scale)), max(1, round(roi.shape[0]*scale))), interpolation=cv2.INTER_NEAREST)
                            tile[105:105+resized.shape[0], :resized.shape[1]] = resized
                            tiles.append(tile)
                            if a['color'] != truth['color'] or a['shape'] != truth['shape']:
                                stem = f'{fid:06d}_{index}_{resolution}'
                                cv2.imwrite(str(failures/(stem+'.png')), roi)
                                dump(failures/(stem+'.json'), dict(truth=truth, attributes=a, stage='MANUAL_ROI_NO_DETECTOR'))
                        for model, found in matchsets.items():
                            overlap_rows = sorted(raw[model], key=lambda r: iou(r['bbox'], truth['bbox_q4']), reverse=True)
                            entry['detections'][model] = dict(strict_head_match=index in found,
                                best_iou=iou(overlap_rows[0]['bbox'], truth['bbox_q4']) if overlap_rows else 0,
                                top_overlap_candidates=overlap_rows[:3])
                            if index in found:
                                det = raw[model][found[index]]
                                for resolution, image in [('q4', q4), ('original', original)]:
                                    box = det['bbox'] if resolution == 'q4' else [v*(sx if n%2 == 0 else sy) for n,v in enumerate(det['bbox'])]
                                    attr = asdict(improved.classify(crop(image, box)))
                                    entry['detections'][model][resolution] = attr
                                    score_attr(counts, f'detected_{model}_{resolution}_', truth, attr)
                            else:
                                counts[model+'_detector_misses'] += 1
                        evaluation.append(entry)
                    diagnostic_rows.extend(evaluation)
                canvas = np.zeros((880, 1920, 3), np.uint8)
                for col, model in enumerate(list(models)[:3]):
                    canvas[35:395, col*640:(col+1)*640] = q4
                    text(canvas, f'{model} FLOAT ONNX f{fid} fresh={inferred} det_age={(fid-age)/fps:.2f}s', (col*640+6, 24), .46)
                    for det in raw[model]:
                        if det['confidence'] < .25: continue
                        x1, y1, x2, y2 = map(round, det['bbox'])
                        cv2.rectangle(canvas, (col*640+x1,35+y1), (col*640+x2,35+y2),(0,210,255),1)
                    for n, resolution in enumerate(('q4', 'original')):
                        valid = [s for s in results[model][resolution]['signals'] if s['valid']]
                        text(canvas, resolution+' 2Hz tracks: '+str(len(valid)), (col*640+8,420+n*190),.5)
                        for j, s in enumerate(valid[:5]):
                            text(canvas, f"#{s['track_id']} {s['color']} {s['shape']} {s['reason']}",
                                 (col*640+8,445+n*190+j*25),.34)
                text(canvas,'MANUAL GT: reviewed frames only. Head IoU>=0.5 is separate from lamp localization. No ego-road binding.',(10,830),.55)
                text(canvas,'Original = source detail upper bound. Q4 = 640x360 deployment-like input. No hardware or board FPS claim.',(10,857),.55)
                writer.write(canvas)
                stream.write(json.dumps(dict(frame_id=fid, timestamp_ms=fid/fps*1000,
                    inference_fresh=inferred, detection_frame_id=age, raw_detections=raw,
                    pipelines=results, reviewed=evaluation), allow_nan=False)+'\n')
                fid += 1
                if fid % 150 == 0: print(clip['id'], fid, '/', total, flush=True)
        cap.release(); writer.release()
        if tiles:
            while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(output/'manual_roi_comparison.jpg'), np.vstack([np.hstack(tiles[n:n+4]) for n in range(0,len(tiles),4)]))
        report = dict(clip=clip['id'], video=clip['path'], sha256=video_hash, frames=fid,
            annotation_scope='REVIEWED_SPARSE_HEADS_ONLY_NOT_EXHAUSTIVE', counts=dict(counts),
            inference_stride=args.stride, attribute_hz=2,
            metrics={key[:-6]+'_accuracy': counts[key[:-6]+'_correct']/value for key,value in counts.items() if key.endswith('_total') and value and key not in ('gt_objects',)},
            association=dict(ego_road_accuracy=None, reason='NO_VERIFIED_HEAD_ROAD_BINDINGS'),
            detector_ms={m:float(np.mean([r['ms'] for r in timing if r['model']==m])) for m in models},
            limitations=['FLOAT_GRAPH_NOT_RKNN_QUANTIZED','SPARSE_GT','NO_FULL_SCENE_PRECISION_WITH_PARTIAL_ANNOTATIONS'])
        dump(output/'report.json', report); dump(output/'diagnostics.json', diagnostic_rows)
        dump(output/'timing.json', timing)
        reports[clip['id']] = report
        print(clip['id'], 'complete', dict(counts), flush=True)
    dump(args.output/'report.json', reports)


if __name__ == '__main__':
    main()
