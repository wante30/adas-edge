"""Run frozen PC detections and experimental events on official front videos.

Event counts are algorithm outputs, not precision/recall or board results.
"""
from collections import Counter
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from competition_ai.hazard_events import HazardEvents
from competition_ai.pc_detector import PcDetector
from competition_ai.scenario import Controls, Mode
from competition_ai.scenario_policy import decide
from tools.competition_console.pipeline import FrontSnapshot
from competition_ai.bound_signal import SignalObservation

MODEL = Path('D:/Datasets/adas_edge_training/rect_models/yolo11n_512x288_split.onnx')
VIDEOS = Path('D:/BaiduNetdiskDownload/紫光同创杯赛题三参考素材')


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def audit(video, detector, output):
    directory = output/video.stem
    directory.mkdir()
    cap = cv2.VideoCapture(str(video))
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps <= 0:
        raise ValueError(f'Cannot decode {video}')
    stride = max(1, round(fps/10))
    writer = cv2.VideoWriter(str(directory/'annotated_mp4v.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps/stride, (960, 540))
    if not writer.isOpened():
        raise ValueError('Cannot create preview')
    engine = HazardEvents()
    counts, guards, detections = Counter(), Counter(), Counter()
    timings, samples, event_ranges = [], 0, []
    started = time.perf_counter()
    try:
        with (directory/'trace.jsonl').open('w', encoding='utf-8') as trace:
            for fid in range(total):
                if fid % stride:
                    if not cap.grab():
                        raise RuntimeError(f'Unexpected source end {fid}')
                    continue
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Unexpected source end {fid}')
                items, elapsed = detector.detect(frame)
                timings.append(elapsed)
                hazard = engine.update(frame, items, fid, round(fid/fps*1e9))
                snapshot = FrontSnapshot(items, [], SignalObservation(), 0., None, 'UNBOUND', hazard)
                decision = decide(Controls(mode=Mode.HAZARD), snapshot.as_perception(), round(fid/fps*1e9))
                trace.write(json.dumps({'frame': fid, 'source_s': fid/fps, 'detections': items,
                                        'hazards': asdict(hazard), 'decision': asdict(decision)}, ensure_ascii=False)+'\n')
                counts.update(e['kind'] for e in hazard.events)
                guards.update([hazard.reason])
                detections.update(d['class_name'] for d in items)
                thumb = cv2.resize(frame, (960, 540))
                cv2.polylines(thumb, [np.int32(np.array(hazard.path)*[960, 540])], True, (240, 180, 60), 2)
                for item in hazard.tracks:
                    x1, y1, x2, y2 = np.int32(np.array(item['bbox'])/np.array([frame.shape[1], frame.shape[0]]*2)*np.array([960, 540]*2))
                    colour = (60, 200, 250) if item['id'] == hazard.lead_id else (180, 180, 180)
                    cv2.rectangle(thumb, (x1, y1), (x2, y2), colour, 1)
                    cv2.putText(thumb, f"{item['class_name']} ID{item['id']}", (x1, max(18, y1-3)), cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1)
                labels = ', '.join(e['kind'] for e in hazard.events) or 'NO EVENT'
                cv2.rectangle(thumb, (0, 0), (960, 89), (25, 25, 25), -1)
                danger = hazard.pedestrian_hazard or hazard.obstacle_hazard
                for y, text, colour in [(21, f'{video.name}  {fid/fps:.2f}s | {decision.maneuver.value.upper()} | {labels}', (70, 90, 255) if danger else (235, 235, 235)),
                                         (44, 'EXPERIMENTAL IMAGE PATH | sudden brake: UNKNOWN | pedestrian red violation: UNKNOWN', (180, 210, 240)),
                                         (69, f'camera: {hazard.reason} | dx={hazard.motion.dx:.3f} scale={hazard.motion.scale_rate:.3f}', (190, 190, 190))]:
                    cv2.putText(thumb, text, (9, y), cv2.FONT_HERSHEY_SIMPLEX, .43, colour, 1)
                if hazard.events:
                    if not event_ranges or fid/fps-event_ranges[-1]['end_s'] > stride/fps+.01:
                        event_ranges.append({'start_s': fid/fps, 'end_s': fid/fps, 'kinds': list({e['kind'] for e in hazard.events})})
                        cv2.imwrite(str(directory/f'event_{fid:05d}.jpg'), thumb)
                    else:
                        event_ranges[-1]['end_s'] = fid/fps
                writer.write(thumb)
                samples += 1
    finally:
        cap.release()
        writer.release()
    report = {'scope': 'PC_FP32_DETECTIONS_IMAGE_EVENT_PROTOTYPE_NOT_RKNN_OR_FPGA',
              'video': str(video), 'video_sha256': digest(video), 'source_frames': total,
              'source_fps': fps, 'analyzed_source_stride': stride, 'analyzed_frames': samples,
              'duration_s': total/fps, 'wall_s': time.perf_counter()-started,
              'event_sample_counts': dict(counts), 'event_ranges': event_ranges,
              'guard_sample_counts': dict(guards), 'detection_sample_counts': dict(detections),
              'postprocess_ms': {'median': float(np.median(engine.timings)), 'p95': float(np.percentile(engine.timings, 95))},
              'main_model_inference_ms': {'median': float(np.median(timings)), 'p95': float(np.percentile(timings, 95))},
              'accuracy_status': 'NO_EVENT_GROUND_TRUTH_METRICS; counts are outputs, not verified event labels',
              'sudden_braking_status': 'UNVERIFIED_NO_EGO_SPEED_OR_CALIBRATED_RANGE',
              'pedestrian_red_violation_status': 'UNVERIFIED_NO_ASSOCIATED_PEDESTRIAN_SIGNAL'}
    (directory/'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        subprocess.run([ffmpeg, '-v', 'error', '-i', str(directory/'annotated_mp4v.mp4'), '-c:v', 'libx264',
                        '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(directory/'annotated.mp4')], check=True)
    print(json.dumps({'video': video.name, 'samples': samples, 'events': dict(counts),
                      'guards': dict(guards), 'postprocess_ms': report['postprocess_ms']}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--videos', type=Path, nargs='+', default=[VIDEOS/f'video_20260929_{n}.mp4' for n in ('112507', '113343', '113421', '113452', '113539')])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a fresh output directory')
    args.output.mkdir(parents=True)
    detector = PcDetector(args.model, (512, 288))
    reports = [audit(video, detector, args.output) for video in args.videos]
    (args.output/'summary.json').write_text(json.dumps({'model': str(args.model), 'model_sha256': digest(args.model),
                                                       'videos': reports}, indent=2, ensure_ascii=False), encoding='utf-8')
