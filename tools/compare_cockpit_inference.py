"""Compare complete cockpit traces without rerunning inference.

This is a numerical/semantic regression check, never an accuracy benchmark.
Source timestamps/ages remain in the comparison; only inference wall timings
are excluded. No reference labels or board execution are implied.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


SUMMARY_CONTRACT = (
    'video_sha256', 'main_sha256', 'specialist_sha256', 'source_fps',
    'requested_source_time_hz', 'same_frame_navigation_enabled', 'navigation_hz',
    'hazards_enabled',
)
ROW_CONTRACT = (
    'frame', 'source_s', 'controls', 'decision', 'signal', 'binding_source',
    'hazards', 'main_age_ms', 'specialist_age_ms', 'navigation',
    'same_frame_split', 'road_inference_frame_id', 'hud_software_receipt',
)


def strip_wall_timings(value):
    if isinstance(value, dict):
        return {k: strip_wall_timings(v) for k, v in value.items()
                if k not in ('inference_ms', 'postprocess_ms')}
    if isinstance(value, list):
        return [strip_wall_timings(v) for v in value]
    return value


def detections(value):
    if not isinstance(value, list):
        raise ValueError('Detection field must be a list')
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get('class_name'), str):
            raise ValueError('Each detection requires class_name')
        bbox, score = item.get('bbox'), item.get('confidence')
        if (not isinstance(bbox, list) or len(bbox) != 4
                or not all(type(v) in (int, float) and math.isfinite(v) for v in bbox)
                or type(score) not in (int, float) or not math.isfinite(score)):
            raise ValueError('Each detection requires four finite coordinates and a finite confidence')
    # Enumeration order alone has no visual/decision meaning. Keep all fields;
    # unknown extra detection semantics must not be silently discarded.
    return sorted(value, key=lambda item: (item['class_name'], *item['bbox'], item['confidence']))


def compare_rows(before, after, *, bbox_atol=0., score_atol=0., detail_limit=40):
    errors = Counter()
    details = []
    maxima = {'bbox_abs_difference': 0., 'score_abs_difference': 0.}

    def difference(section, context, a, b):
        errors[section] += 1
        if len(details) < detail_limit:
            details.append({'section': section, 'context': context, 'before': a, 'after': b})

    def walk(a, b, section, context):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                difference(section, context+'.keys', sorted(a), sorted(b))
            for key in sorted(a.keys() & b.keys()):
                walk(a[key], b[key], section, context+'.'+key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                difference(section, context+'.length', len(a), len(b))
            for index, (left, right) in enumerate(zip(a, b)):
                walk(left, right, section, f'{context}[{index}]')
        elif type(a) in (int, float) and type(b) in (int, float):
            delta = abs(a-b)
            tolerance = 0.
            if section in ('main', 'specialist') and '.bbox[' in context:
                tolerance = bbox_atol
                maxima['bbox_abs_difference'] = max(maxima['bbox_abs_difference'], delta)
            elif section in ('main', 'specialist') and context.endswith('.confidence'):
                tolerance = score_atol
                maxima['score_abs_difference'] = max(maxima['score_abs_difference'], delta)
            if not math.isfinite(a) or not math.isfinite(b) or delta > tolerance:
                difference(section, context, a, b)
        elif type(a) is not type(b) or a != b:
            difference(section, context, a, b)

    if len(before) != len(after):
        difference('coverage', 'trace_row_count', len(before), len(after))
    for index, (left, right) in enumerate(zip(before, after)):
        context = f'row={index},frame={left.get("frame")}'
        for field in ROW_CONTRACT:
            if field not in left or field not in right:
                difference('schema', context+'.'+field, field in left, field in right)
            else:
                walk(strip_wall_timings(left[field]), strip_wall_timings(right[field]), field, context+'.'+field)
        for field in ('main', 'specialist'):
            try:
                a, b = detections(left.get(field)), detections(right.get(field))
            except ValueError as exc:
                difference('schema', context+'.'+field, 'complete detection evidence required', str(exc))
                continue
            walk(a, b, field, context+'.'+field)
        for side, row in (('before', left), ('after', right)):
            receipt = row.get('hud_software_receipt', {})
            if receipt.get('receiver_verified') is not True or receipt.get('hardware_writes') != 0:
                difference('hud_integrity', context+'.'+side, 'verified software receipt / zero hardware writes',
                           {'receiver_verified': receipt.get('receiver_verified'),
                            'hardware_writes': receipt.get('hardware_writes')})
    return {'pass': not errors, 'difference_counts': dict(errors), 'differences': details,
            'detail_limit': detail_limit, **maxima}


def inventory(rows):
    epoch, previous = 0, None
    frames, main, specialist, navigation = set(), set(), set(), set()
    detections_by_class = {'main': Counter(), 'specialist': Counter()}
    for row in rows:
        frame = row['frame']
        if previous is not None and frame < previous:
            epoch += 1
        key = (epoch, frame)
        frames.add(key)
        for name, ids in (('main', main), ('specialist', specialist)):
            if row.get(name+'_age_ms') == 0 and key not in ids:
                ids.add(key)
                detections_by_class[name].update(d['class_name'] for d in detections(row.get(name)))
        source = row.get('navigation', {}).get('source_frame_id')
        if source is not None:
            navigation.add((epoch, source))
        previous = frame
    ids = sorted(frame for e, frame in frames if e == 0)
    return {'trace_rows': len(rows), 'unique_source_frames': len(frames), 'replays': epoch,
            'first_frame': ids[0] if ids else None, 'last_frame': ids[-1] if ids else None,
            'first_epoch_contiguous': ids == list(range(ids[0], ids[-1]+1)) if ids else False,
            'main_updates': len(main), 'specialist_updates': len(specialist),
            'navigation_updates': len(navigation),
            'detections_by_class_at_updates': {k: dict(v) for k, v in detections_by_class.items()}}


def load_run(path):
    directory = Path(path).resolve()
    trace = directory/'trace.jsonl'
    with trace.open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f'Empty trace: {trace}')
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    with trace.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return rows, summary, {'directory': str(directory), 'trace_sha256': digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True, type=Path)
    parser.add_argument('--after', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expected-frames', type=int)
    parser.add_argument('--expected-main', type=int)
    parser.add_argument('--expected-specialist', type=int)
    parser.add_argument('--expected-navigation', type=int)
    parser.add_argument('--bbox-atol', type=float, default=0.)
    parser.add_argument('--score-atol', type=float, default=0.)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v < 0 for v in (args.bbox_atol, args.score_atol)):
        parser.error('Tolerances must be finite and nonnegative')
    target = args.output/'comparison.json'
    if target.exists():
        raise ValueError('Choose a fresh output directory; previous evidence is preserved')
    before, bs, bp = load_run(args.before)
    after, ass, ap = load_run(args.after)
    result = compare_rows(before, after, bbox_atol=args.bbox_atol, score_atol=args.score_atol)
    metadata = []
    for key in SUMMARY_CONTRACT:
        if key not in bs or key not in ass or bs.get(key) != ass.get(key):
            metadata.append({'field': key, 'before': bs.get(key), 'after': ass.get(key)})
    bi, ai = inventory(before), inventory(after)
    coverage = []
    for side, inv, summary in (('before', bi, bs), ('after', ai, ass)):
        for name, expected in (('unique_source_frames', args.expected_frames), ('main_updates', args.expected_main),
                               ('specialist_updates', args.expected_specialist), ('navigation_updates', args.expected_navigation)):
            if expected is not None and inv[name] != expected:
                coverage.append({'side': side, 'field': name, 'expected': expected, 'actual': inv[name]})
        for name in ('main', 'specialist', 'navigation'):
            if inv[name+'_updates'] != summary.get(name+'_inference_updates'):
                coverage.append({'side': side, 'field': name+'_updates', 'trace': inv[name+'_updates'],
                                 'summary': summary.get(name+'_inference_updates')})
        if not inv['first_epoch_contiguous']:
            coverage.append({'side': side, 'field': 'source_frames', 'reason': 'non_contiguous'})
    result.update(schema='cockpit_inference_regression.v1', before=bp, after=ap,
                  before_inventory=bi, after_inventory=ai, metadata_differences=metadata,
                  coverage_errors=coverage, tolerances={'bbox_atol': args.bbox_atol, 'score_atol': args.score_atol},
                  self_comparison=args.before.resolve() == args.after.resolve(),
                  excluded_wall_timing_fields=['inference_ms', 'postprocess_ms'],
                  scope='PC_OUTPUT_REGRESSION_NOT_ABSOLUTE_ACCURACY_OR_RK3566_VALIDATION',
                  limitations=['Equal outputs can contain the same false positives and misses.',
                               'No ground truth, precision/recall, NPU quantization or board speed is evaluated.',
                               'This source has unassociated signals; control equivalence does not validate green/red branches.',
                               'Navigation reading is specific to the calibrated video layout.'])
    result['pass'] = result['pass'] and not metadata and not coverage
    args.output.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'pass': result['pass'], 'self_comparison': result['self_comparison'],
                      'trace_frames': ai['unique_source_frames'], 'main': ai['main_updates'],
                      'specialist': ai['specialist_updates'], 'navigation': ai['navigation_updates'],
                      'report': str(target.resolve())}, ensure_ascii=False))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
