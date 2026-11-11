"""Separate visual arrow direction from evidence of ego-road relevance.

An external binding is a time-limited calibrated head/road observation, not
the raw FPGA lane-marking type or a navigation instruction. No spatial-order
guessing or default round-lamp lane assignment is allowed.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .multi_signal import DIRECTIONS, iou, unknown_directions

ARROWS = {'LEFT_ARROW': 'left', 'STRAIGHT_ARROW': 'straight', 'RIGHT_ARROW': 'right', 'UTURN_ARROW': 'uturn'}


def associate(result, *, now_ns, bindings=(), navigation_intent=None, copy_result=True):
    out = deepcopy(result) if copy_result else result
    for signal in out['signals']:
        for key in ('associated_directions', 'association_confidence', 'association_expires_at_ns', 'head_id', 'group_id'):
            signal.pop(key, None)
        signal['direction'] = ARROWS.get(signal['shape'])
        signal.update(association='UNASSOCIATED', ego_relevant='UNKNOWN',
                      association_reason='NO_HEAD_ROAD_BINDING', association_evidence=[])
        if (not signal['timestamp_ns'] <= now_ns < signal['expires_at_ns']
                or not signal['valid']) or signal['color'] == 'UNKNOWN' or signal['shape'] == 'UNKNOWN':
            signal['association_reason'] = signal['reason']
            continue
        matches = []
        for binding in bindings:
            try:
                confidence = float(binding['confidence'])
                identity = (binding['source_id'] == 'Q4' and binding['context'] == out['context']
                            and binding['coordinate_space'] == 'q4_native'
                            and bool(binding['head_id']) and bool(binding['group_id'])
                            and bool(binding['calibration_id']))
                times = (0 <= now_ns-binding['timestamp_ns'] < 500_000_000
                         and now_ns < binding['expires_at_ns']
                         and binding['expires_at_ns']-binding['timestamp_ns'] <= 500_000_000)
                evidence = (binding['evidence_type'] in ('CALIBRATED_VISUAL_HEAD', 'FPGA_CALIBRATED_HEAD')
                            and binding.get('ego_road_confirmed') is True
                            and binding.get('viewpoint_confirmed') is True)
                if not identity or not times or not evidence or not math.isfinite(confidence) or not .8 <= confidence <= 1:
                    continue
                raw = binding['bbox']
                if (len(raw) != 4 or not all(math.isfinite(float(v)) for v in raw)
                        or max(iou(raw, signal['bbox']), iou(raw, signal.get('parent_head_bbox', signal['bbox']))) < .5):
                    continue
                allowed = binding['directions']
                if not allowed or any(d not in DIRECTIONS for d in allowed):
                    continue
                matches.append(binding)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        # Multiple candidate heads/roads may never silently select the first.
        identities = {(b['head_id'], b['group_id'], tuple(sorted(b['directions']))) for b in matches}
        if len(identities) > 1:
            signal['association_reason'] = 'AMBIGUOUS_HEAD_ROAD_BINDING'
            continue
        if not matches:
            if signal['shape'] == 'ROUND':
                signal['association_reason'] = 'ROUND_WITHOUT_EXPLICIT_LANE_EVIDENCE'
            continue
        binding = max(matches, key=lambda b: b['confidence'])
        direction = signal['direction']
        if direction and direction not in binding['directions']:
            signal['association_reason'] = 'ARROW_BINDING_DISAGREEMENT'
            continue
        signal.update(association='ASSOCIATED', ego_relevant='YES',
                      associated_directions=[direction] if direction else binding['directions'],
                      association_confidence=min(binding['confidence'], signal['confidence'],
                          signal['color_confidence'], signal['shape_confidence']),
                      association_expires_at_ns=min(signal['expires_at_ns'], binding['expires_at_ns']),
                      head_id=binding['head_id'], group_id=binding['group_id'],
                      association_reason='CALIBRATED_HEAD_ROAD_EVIDENCE',
                      association_evidence=[dict(binding)])
    out['directions'] = unknown_directions('NO_ASSOCIATED_SIGNAL')
    for direction in DIRECTIONS:
        signals = [s for s in out['signals'] if s.get('association') == 'ASSOCIATED'
                   and direction in s.get('associated_directions', [])]
        if not signals:
            continue
        colors = {s['color'] for s in signals}
        ids = [s['track_id'] for s in signals]
        groups = sorted({s['group_id'] for s in signals})
        if len(colors) != 1:
            out['directions'][direction].update(reason='CONFLICTING_ASSOCIATED_COLORS',
                signal_ids=ids, groups=groups, conflict_colors=sorted(colors))
        else:
            out['directions'][direction] = dict(color=next(iter(colors)), valid=True,
                confidence=min(s['association_confidence'] for s in signals),
                reason='ASSOCIATED_SIGNALS_AGREE', signal_ids=ids, groups=groups,
                timestamp_ns=min(s['timestamp_ns'] for s in signals),
                expires_at_ns=min(s['association_expires_at_ns'] for s in signals))
    out['navigation_intent'] = navigation_intent  # context only; never used above
    # Descriptive visual summary, explicitly separate from ego-road passage
    # signals above. It may include other roads and grants no relevance.
    out['observed_directions'] = unknown_directions('NO_CONFIRMED_VISUAL_ARROW')
    for direction in DIRECTIONS:
        rows = [s for s in out['signals'] if s.get('valid') and s.get('direction') == direction
                and s['color'] != 'UNKNOWN' and s['timestamp_ns'] <= now_ns < s['expires_at_ns']]
        if not rows:
            continue
        colors = {s['color'] for s in rows}
        common = dict(signal_ids=[s['track_id'] for s in rows], ego_relevant='UNKNOWN',
                      timestamp_ns=min(s['timestamp_ns'] for s in rows),
                      expires_at_ns=min(s['expires_at_ns'] for s in rows))
        if len(colors) > 1:
            out['observed_directions'][direction].update(common, reason='CONFLICTING_VISIBLE_ARROWS',
                                                        conflict_colors=sorted(colors))
        else:
            out['observed_directions'][direction] = dict(common, color=next(iter(colors)), valid=True,
                confidence=min(min(s['confidence'], s['color_confidence'], s['shape_confidence']) for s in rows),
                reason='VISUAL_ARROW_ONLY_EGO_ROAD_UNCONFIRMED')
    return out
