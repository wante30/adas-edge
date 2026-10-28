"""Isolated PC navigation profiling; default mode reads existing evidence only.

The experimental recognition-only wrapper is confined to this benchmark. It
does not modify production reader/session logic, weights, pixels, thresholds,
or the 3 Hz sampling schedule. Run inference only in an agreed exclusive CPU
window. Equality with a baseline is a regression check, not ground truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_AUDIT = ROOT/'outputs/navigation_112739_20261016/reader'
DEFAULT_RECORDING = ROOT/'outputs/navigation_112739_20261016/recording'
DEFAULT_OUTPUT = ROOT/'outputs/cockpit_perf_20261016/navigation'
SEMANTIC_KEYS = ('source_frame_id', 'timestamp_ns', 'valid', 'direction',
                 'distance_m', 'road_name', 'reason', 'raw_text')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compare_observations(reference, candidate):
    """Compare values and exact same-frame inputs independently of runtime."""
    left = {row['source_frame_id']: row for row in reference}
    right = {row['source_frame_id']: row for row in candidate}
    missing = sorted(set(left)-set(right))
    extra = sorted(set(right)-set(left))
    semantic, confidence, inputs, tensors = [], [], [], []
    input_compared, tensor_compared = 0, 0
    for fid in sorted(set(left)&set(right)):
        a, b = left[fid], right[fid]
        changes = {key: [a.get(key), b.get(key)] for key in SEMANTIC_KEYS if a.get(key) != b.get(key)}
        if changes:
            semantic.append({'frame_id': fid, 'changes': changes})
        if a.get('confidence') != b.get('confidence'):
            confidence.append({'frame_id': fid, 'before': a.get('confidence'), 'after': b.get('confidence')})
        if a.get('input_sha256') and b.get('input_sha256'):
            input_compared += 1
            if a['input_sha256'] != b['input_sha256']:
                inputs.append(fid)
        if a.get('tensor_digests') is not None and b.get('tensor_digests') is not None:
            tensor_compared += 1
            if a['tensor_digests'] != b['tensor_digests']:
                tensors.append(fid)
    return {'missing_frames': missing, 'extra_frames': extra,
            'semantic_differences': semantic, 'confidence_exact_differences': confidence,
            'input_pixel_differences': inputs, 'tensor_digest_differences': tensors,
            'input_digest_compared_frames': input_compared,
            'tensor_digest_compared_frames': tensor_compared,
            'regression_equal': not (missing or extra or semantic or confidence or inputs or tensors),
            'accuracy_claim': 'NONE_OUTPUT_EQUALITY_IS_NOT_GROUND_TRUTH'}


def reviewed_labels(audit_dir=DEFAULT_AUDIT):
    """Reuse explicitly human-reviewed cases, never model outputs as labels.

    Frame IDs select evaluation annotations only. They are never passed to the
    recognizer as an answer schedule. This is a development regression subset.
    """
    labels = {}
    with (Path(audit_dir)/'manual_error_case_review.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            labels[int(row['frame_id'])] = {
                'direction': row['visually_read_direction'],
                'distance_m': float(row['visually_read_distance_m']),
                'provenance': 'MANUAL_ERROR_CASE_REVIEW_20261016',
            }
    for fid in [20, 30, 40, *range(2200, 2520, 10)]:
        labels[fid] = {'direction': 'none', 'distance_m': None,
                       'provenance': 'MANUALLY_REVIEWED_SIDE_VIEW_OR_TRANSITION_SAMPLES'}
    for fid, direction, distance in [(0, 'right', 893.), (1500, 'left', 461.), (2730, 'uturn', 195.)]:
        labels[fid] = {'direction': direction, 'distance_m': distance,
                       'provenance': 'MANUALLY_REVIEWED_NAVIGATION_POSITIVE_20261016'}
    return labels


def evaluate_reviewed(rows, labels):
    by_frame = {row['source_frame_id']: row for row in rows}
    counts = {'positive_labels': 0, 'positive_exact': 0, 'positive_rejected': 0,
              'positive_wrong_accepted': 0, 'negative_labels': 0,
              'negative_correct_rejection': 0, 'negative_false_accepted': 0}
    failures, missing = [], []
    for fid, label in labels.items():
        if fid not in by_frame:
            missing.append(fid)
            continue
        row = by_frame[fid]
        if label['direction'] == 'none':
            counts['negative_labels'] += 1
            key = 'negative_false_accepted' if row['valid'] else 'negative_correct_rejection'
        else:
            counts['positive_labels'] += 1
            key = ('positive_rejected' if not row['valid'] else
                   'positive_exact' if (row['direction'], row['distance_m']) ==
                                       (label['direction'], label['distance_m']) else 'positive_wrong_accepted')
        counts[key] += 1
        if key in ('positive_wrong_accepted', 'negative_false_accepted'):
            failures.append({'frame_id': fid, 'label': label,
                             'prediction': {key: row.get(key) for key in SEMANTIC_KEYS}})
    return {**counts, 'missing_reviewed_frames': sorted(missing), 'wrong_acceptances': failures,
            'scope': '45_DEVELOPMENT_REVIEW_SAMPLES_NOT_INDEPENDENT_GENERALIZATION_BENCHMARK',
            'all_rejections_are_success': False}


def inspect_existing(recording_dir=DEFAULT_RECORDING, audit_dir=DEFAULT_AUDIT):
    summary = json.loads((Path(recording_dir)/'summary.json').read_text(encoding='utf-8'))
    observations = {}
    for row in read_jsonl(Path(recording_dir)/'trace.jsonl'):
        nav = row.get('navigation')
        if nav and nav.get('source_frame_id') is not None:
            observations.setdefault(nav['source_frame_id'], nav)
    rows = list(observations.values())
    durations = [row['inference_ms'] for row in rows]
    total = sum(durations)/1000
    return {'evidence': 'READ_EXISTING_TRACE_NO_NEW_INFERENCE',
            'recording_wall_s': summary['wall_seconds'], 'updates': len(rows),
            'navigation_update_total_s': total,
            'navigation_direct_share_pct': total/summary['wall_seconds']*100,
            'first_update_ms': durations[0], 'warm_update_total_s': sum(durations[1:])/1000,
            'warm_update_median_ms': statistics.median(durations[1:]),
            'zero_navigation_cost_optimistic_wall_s': summary['wall_seconds']-total,
            'limit': 'Excludes unmeasured background thread contention; not an end-to-end speedup prediction.',
            'audit_regression': compare_observations(read_jsonl(Path(audit_dir)/'sampling_3hz.jsonl'), rows),
            'reviewed_accuracy': evaluate_reviewed(rows, reviewed_labels(audit_dir))}


class RecognitionOnlyEngine:
    """Experimental 3.9.2 adapter retaining upstream preprocessing/recognizer.

    It uses an upstream private configuration loader and is therefore version
    pinned and benchmark-only. Production adoption requires explicit review,
    tensor/observation equivalence and offline tests. No detector/classifier
    sessions are constructed. Model/character dictionary are unchanged.
    """
    def __init__(self):
        import importlib.metadata
        if importlib.metadata.version('rapidocr') != '3.9.2':
            raise RuntimeError('Experimental wrapper is pinned to rapidocr==3.9.2')
        from rapidocr import RapidOCR
        from rapidocr.ch_ppocr_rec import TextRecognizer
        from rapidocr.utils.load_image import LoadImage
        config_loader = RapidOCR.__new__(RapidOCR)
        cfg = config_loader._load_config(None, {
            'Global.log_level': 'warning', 'Global.use_det': False, 'Global.use_cls': False,
            'EngineConfig.onnxruntime.intra_op_num_threads': 2,
            'EngineConfig.onnxruntime.inter_op_num_threads': 1,
        })
        cfg.Rec.engine_cfg = cfg.EngineConfig[cfg.Rec.engine_type.value]
        cfg.Rec.font_path = cfg.Global.font_path
        cfg.Rec.model_root_dir = cfg.Global.model_root_dir
        self.cfg, self.load_img = cfg, LoadImage()
        self.text_rec = TextRecognizer(cfg.Rec)

    def __call__(self, image, *, use_det=False, use_cls=False):
        if use_det or use_cls:
            raise ValueError('Recognition-only benchmark cannot execute detection/classification')
        from rapidocr.ch_ppocr_rec import TextRecInput
        from rapidocr.utils.process_img import resize_image_within_bounds
        image = self.load_img(image)
        if self.cfg.Global.use_preprocess_img:
            image, _, _ = resize_image_within_bounds(
                image, self.cfg.Global.min_side_len, self.cfg.Global.max_side_len)
        return self.text_rec(TextRecInput(img=[image], return_word_box=False))


class TimedCall:
    def __init__(self, original, timings, tensor_digests=None):
        self.original, self.timings, self.tensor_digests = original, timings, tensor_digests

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        result = self.original(*args, **kwargs)
        self.timings.append((time.perf_counter()-start)*1000)
        if self.tensor_digests is not None:
            # Instrumented runs are accuracy evidence only; hashing overhead
            # must not be mistaken for production runtime.
            self.tensor_digests.append({'input': hashlib.sha256(args[0].tobytes()).hexdigest(),
                                        'output': hashlib.sha256(result.tobytes()).hexdigest()})
        return result

    def __getattr__(self, key):
        return getattr(self.original, key)


def run_benchmark(profile_path, output_dir, variant, *, tensor_digests=False):
    import cv2
    import numpy as np
    from tools.competition_console.navigation_reader import NavigationReader
    from tools.competition_console.split_video import SplitVideo

    class ProfiledReader(NavigationReader):
        def __init__(self):
            super().__init__()
            self.init_ms = None
            self.arrow_ms, self.ocr_ms, self.rec_ms, self.ort_ms = [], [], [], []
            self.digests = [] if tensor_digests else None

        def _engine(self):
            if self._ocr is None:
                start = time.perf_counter()
                engine = (RecognitionOnlyEngine() if variant == 'recognition-only'
                          else super()._engine())
                self.init_ms = (time.perf_counter()-start)*1000
                engine.text_rec.session = TimedCall(engine.text_rec.session, self.ort_ms, self.digests)
                engine.text_rec = TimedCall(engine.text_rec, self.rec_ms)
                self._ocr = TimedCall(engine, self.ocr_ms)
            return self._ocr

        def _arrow(self, image):
            start = time.perf_counter()
            result = super()._arrow(image)
            self.arrow_ms.append((time.perf_counter()-start)*1000)
            return result

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir/'observations.jsonl'
    if rows_path.exists():
        raise ValueError('Use a fresh output directory; previous evidence is preserved')
    profile_path = Path(profile_path)
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    if sha256(profile['video']) != profile['video_sha256']:
        raise ValueError('Source video hash mismatch')
    reader, splitter = ProfiledReader(), SplitVideo(profile)
    capture = cv2.VideoCapture(profile['video'])
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps != 30.:
        raise ValueError('This comparison requires the reviewed 30 FPS input')
    rows, fid = [], 0
    start = time.perf_counter()
    try:
        with rows_path.open('w', encoding='utf-8') as stream:
            while True:
                ok, source = capture.read()
                if not ok:
                    break
                if fid % 10 == 0:  # exactly 3 Hz, unchanged from production
                    nav = splitter.split(source, fid).navigation
                    begin = len(reader.digests or [])
                    row = reader.update(nav, fid, round(fid/fps*1e9)).to_dict()
                    row['input_sha256'] = hashlib.sha256(nav.tobytes()).hexdigest()
                    if tensor_digests:
                        row['tensor_digests'] = reader.digests[begin:]
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                fid += 1
    finally:
        capture.release()
    def describe(values):
        return {'calls': len(values), 'total_ms': sum(values),
                'median_ms': float(np.median(values)) if values else None,
                'p95_ms': float(np.percentile(values, 95)) if values else None}
    report = {'evidence': 'PC_COMPONENT_BENCHMARK_NOT_RK3566_OR_FPGA', 'variant': variant,
              'instrumented_for_tensor_equivalence': tensor_digests,
              'timing_eligible': not tensor_digests,
              'wall_s_including_decode_split_and_evidence': time.perf_counter()-start,
              'frames_decoded': fid, 'samples': len(rows), 'source_hz': 3.,
              'engine_initialization_ms': reader.init_ms,
              'stages_nested_not_additive': {
                  'reader_update': describe([r['inference_ms'] for r in rows]),
                  'arrow_and_digit_geometry': describe(reader.arrow_ms),
                  'ocr_wrapper': describe(reader.ocr_ms), 'recognizer': describe(reader.rec_ms),
                  'onnx_runtime': describe(reader.ort_ms)},
              'reader_sha256': sha256(ROOT/'tools/competition_console/navigation_reader.py'),
              'session_sha256': sha256(ROOT/'tools/competition_console/navigation_session.py'),
              'profile_sha256': sha256(profile_path), 'video_sha256': profile['video_sha256'],
              'model_sha256': sha256(ROOT/'.venv/Lib/site-packages/rapidocr/models/PP-OCRv6_rec_small.onnx'),
              'reviewed_accuracy': evaluate_reviewed(rows, reviewed_labels()),
              'reviewed_audit_comparison': compare_observations(read_jsonl(DEFAULT_AUDIT/'sampling_3hz.jsonl'), rows),
              'limits': ['No independent full-frame ground truth; equivalence is not accuracy.',
                         'Pixel and tensor hashing runs must not be used as end-to-end performance.',
                         'No RK3566 memory, FPS, NPU, PCIe or FPGA evidence.']}
    (output_dir/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['inspect', 'run', 'compare'], default='inspect')
    parser.add_argument('--variant', choices=['baseline', 'recognition-only'], default='baseline')
    parser.add_argument('--profile', type=Path, default=ROOT/'configs/competition_native_navigation_112739.json')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--tensor-digests', action='store_true')
    args = parser.parse_args()
    if args.mode == 'run':
        report = run_benchmark(args.profile, args.output, args.variant, tensor_digests=args.tensor_digests)
    elif args.mode == 'compare':
        if not args.reference or not args.candidate:
            parser.error('--reference and --candidate are required for compare')
        report = compare_observations(read_jsonl(args.reference), read_jsonl(args.candidate))
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        report = inspect_existing()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'existing_trace_analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
