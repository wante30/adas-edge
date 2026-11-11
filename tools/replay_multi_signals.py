"""Replay real video + cached detector outputs; never load another detector.

Input is the existing audit's summary.json and raw_predictions.jsonl. Labels
are independent, manually reviewed per-frame objects in native Q4 pixels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from competition_ai.multi_signal import MultiSignalTracker, candidates_from_rows, iou
from competition_ai.multi_signal_worker import expire_result
from competition_ai.signal_shape import ContourTemplateBackend
from competition_ai.signal_association import associate
from competition_ai.traffic_light import classify_traffic_lights


def score_frames(predictions, labels):
    counts = Counter()
    for label in labels:
        fid = label['frame_id']
        predicted = predictions[fid]['signals']
        used = set()
        for truth in label.get('objects', []):
            counts['gt_objects'] += 1
            choices = [(iou(p['bbox'], truth['bbox']), n, p) for n, p in enumerate(predicted) if n not in used]
            match = max(choices, key=lambda item: item[0]) if choices else None
            if not match or match[0] < .5:
                continue
            _, n, observation = match
            used.add(n)
            counts['matched_objects'] += 1
            for key in ('color', 'shape'):
                if truth.get(key) and truth[key] != 'UNKNOWN':
                    counts[key+'_total'] += 1
                    counts[key+'_correct'] += observation[key] == truth[key]
        for direction, wanted in label.get('directions', {}).items():
            if wanted == 'UNKNOWN':
                counts['unknown_direction_total'] += 1
                counts['unknown_direction_correct'] += predictions[fid]['directions'][direction]['color'] == wanted
            else:
                counts['association_total'] += 1
                counts['association_correct'] += predictions[fid]['directions'][direction]['color'] == wanted
    def metric(correct, total):
        n = counts[total]
        return dict(correct=counts[correct], total=n, accuracy=counts[correct]/n if n else None)
    return dict(detection_recall=metric('matched_objects', 'gt_objects'),
                color_accuracy_matched=metric('color_correct', 'color_total'),
                arrow_shape_accuracy_matched=metric('shape_correct', 'shape_total'),
                direction_association_accuracy=metric('association_correct', 'association_total'),
                unknown_direction_correctness=metric('unknown_direction_correct', 'unknown_direction_total'),
                scope='MANUALLY_REVIEWED_SPARSE_FRAMES_NOT_FULL_VIDEO_GT')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    failures = args.output_dir/'failures'; failures.mkdir()
    summary = json.loads((args.input_dir/'summary.json').read_text(encoding='utf-8'))
    cached = {r['frame']: r for r in map(json.loads, (args.input_dir/'raw_predictions.jsonl').read_text(encoding='utf-8').splitlines())}
    cap = cv2.VideoCapture(summary['video'])
    if not cap.isOpened():
        raise RuntimeError('Cannot open original real video '+summary['video'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    context = ('OFFLINE_CACHED_PC_DETECTOR', 1)
    tracker = MultiSignalTracker(backend=ContourTemplateBackend())
    writer = cv2.VideoWriter(str(args.output_dir/'comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 440))
    if not writer.isOpened():
        raise RuntimeError('Video encoder unavailable')
    results, timings, reasons = {}, [], Counter()
    fid, next_update, failure_count = 0, 0, 0
    previous = 'UNKNOWN'
    with (args.output_dir/'frames.jsonl').open('x', encoding='utf-8') as output:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            h, w = image.shape[:2]
            native = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_AREA)
            ns = 1_000_000_000+round(fid/fps*1e9)
            rows = cached.get(fid, {})
            if fid >= next_update:
                candidate_list, detected_rows = [], []
                for label in ('main', 'specialist'):
                    scaled = [dict(row, bbox=[v*(args.width/w if n%2 == 0 else args.height/h)
                                              for n, v in enumerate(row['bbox'])]) for row in rows.get(label, [])]
                    candidates = candidates_from_rows(scaled, frame_id=fid, timestamp_ns=ns,
                        space='q4_native', width=args.width, height=args.height, detector=label)
                    candidate_list.extend(candidates)
                    detected_rows.extend(dict(bbox=c.bbox, confidence=c.confidence) for c in candidates)
                start = time.perf_counter_ns()
                tracker.update(native, candidate_list, frame_id=fid, timestamp_ns=ns, context=context)
                timings.append((time.perf_counter_ns()-start)/1e6)
                previous = classify_traffic_lights(native, detected_rows).value
                next_update = fid+max(1, round(fps/2))
                for track in tracker.tracks:
                    reasons[track.attributes.reason] += 1
                    if track.attributes.shape == 'UNKNOWN' and failure_count < 64:
                        x1, y1, x2, y2 = map(round, track.bbox)
                        roi = native[y1:y2, x1:x2]
                        if roi.size:
                            name = f'{fid:06d}_{track.track_id}_{track.attributes.reason}.png'
                            cv2.imwrite(str(failures/name), roi)
                            failure_count += 1
            result = associate(expire_result(tracker.snapshot(ns), ns), now_ns=ns)
            result['observation_frame_id'] = result['frame_id']
            result['frame_id'] = fid
            result['video_timestamp_ms'] = fid/fps*1000
            results[fid] = result
            output.write(json.dumps(result, allow_nan=False)+'\n')
            before = cv2.resize(native, (640, 360))
            after = before.copy()
            for signal in result['signals']:
                x1, y1, x2, y2 = [round(v*(640/args.width if n%2 == 0 else 360/args.height)) for n, v in enumerate(signal['bbox'])]
                color = {'RED': (40, 40, 255), 'GREEN': (40, 230, 40), 'YELLOW': (0, 230, 230)}.get(signal['color'], (150, 150, 150))
                cv2.rectangle(after, (x1, y1), (x2, y2), color, 1)
                cv2.putText(after, f"#{signal['track_id']} {signal['color']} {signal['shape']}",
                            (max(0, x1), max(14, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .31, color, 1)
            canvas = np.zeros((440, 1280, 3), np.uint8)
            canvas[:360, :640], canvas[:360, 640:] = before, after
            cv2.putText(canvas, f'Legacy global: {previous} (color only)', (8, 385), cv2.FONT_HERSHEY_SIMPLEX, .55, (220, 220, 220), 1)
            cv2.putText(canvas, f'Multi Q4 frame={fid} | L/S/R: UNKNOWN (no road evidence)', (648, 385), cv2.FONT_HERSHEY_SIMPLEX, .45, (220, 220, 220), 1)
            cv2.putText(canvas, 'REAL VIDEO / CACHED PC DETECTIONS / 2 Hz ATTRIBUTES / NOT A BOARD FPS MEASUREMENT',
                        (8, 418), cv2.FONT_HERSHEY_SIMPLEX, .43, (180, 210, 230), 1)
            writer.write(canvas)
            fid += 1
    cap.release(); writer.release()
    labels = json.loads(args.labels.read_text(encoding='utf-8'))['frames'] if args.labels else []
    report = dict(video=summary['video'], source_video_sha256=hashlib.sha256(Path(summary['video']).read_bytes()).hexdigest(),
                  detector_evidence=summary, native_size=[args.width, args.height], frames=fid,
                  attribute_updates=len(timings), attribute_ms_mean=float(np.mean(timings)),
                  attribute_ms_p95=float(np.percentile(timings, 95)), unknown_reasons=dict(reasons),
                  metrics=score_frames(results, labels),
                  untested=['simultaneous_different_direction_colors', 'yellow', 'real_direction_ground_truth',
                            'occlusion_ground_truth', 'RK3566_same_video_AB'])
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'detector_evidence'}, indent=2))


if __name__ == '__main__':
    main()
