"""Source-clock navigation sampling and conservative policy adaptation."""
from dataclasses import asdict, replace

from competition_ai.scenario import Direction


class NavigationSession:
    def __init__(self, reader, *, hz=3.0, valid_for_ms=750):
        if hz <= 0 or not 1 <= valid_for_ms <= 60000:
            raise ValueError('Invalid navigation cadence or lifetime')
        self.reader, self.hz, self.valid_for_ms = reader, hz, valid_for_ms
        self.timings = []
        self.reset()

    def reset(self):
        self.observation = None
        self.last_frame = None
        self.last_update_ns = None
        if hasattr(self.reader, 'reset'):
            self.reader.reset()

    def advance(self, image, frame_id, timestamp_ns):
        if self.last_frame is not None and frame_id < self.last_frame:
            self.reset()
        if frame_id == self.last_frame:
            return
        if self.last_update_ns is None or timestamp_ns-self.last_update_ns >= round(1e9/self.hz)-1:
            result = self.reader.update(image, frame_id, timestamp_ns)
            self.observation = asdict(result)
            if (self.observation.get('source_frame_id') != frame_id
                    or self.observation.get('timestamp_ns') != timestamp_ns):
                self.observation.update(valid=False, direction='none', distance_m=None,
                                        reason='navigation_source_mismatch')
            self.timings.append(float(self.observation.get('inference_ms', 0.0)))
            self.last_update_ns = timestamp_ns
        self.last_frame = frame_id

    def current(self, now_ns, *, ended=False):
        row = dict(self.observation or {
            'direction': 'none', 'distance_m': None, 'road_name': '', 'valid': False,
            'confidence': 0.0, 'reason': 'navigation_not_sampled',
            'source_frame_id': None, 'timestamp_ns': None, 'raw_text': [], 'inference_ms': 0.0,
        })
        stamp = row.get('timestamp_ns')
        age = None if stamp is None else (now_ns-stamp)/1e6
        fresh = age is not None and 0 <= age < self.valid_for_ms
        if ended or not fresh:
            row.update(valid=False, direction='none', distance_m=None, road_name='',
                       reason='source_ended' if ended else 'navigation_expired_or_missing')
        elif not row.get('valid'):
            row.update(direction='none', distance_m=None, road_name='')
        row.update(age_ms=age, valid_for_ms=self.valid_for_ms,
                   evidence='PC_NAVIGATION_SCREEN_PIXELS',
                   policy_supported=row.get('direction') in ('left', 'right', 'forward'))
        return row


def apply_navigation(controls, observation):
    """A route hint never requests a turn or bypasses the traffic-light gate.

    U-turn remains visible in the UI but is not encoded as a left turn in a
    policy/protocol that does not define U-turn semantics.
    """
    direction = observation.get('direction', 'none')
    supported = direction in ('left', 'right', 'forward')
    valid = bool(observation.get('valid')) and supported
    return replace(controls, navigation=Direction(direction) if valid else Direction.NONE,
                   navigation_valid=valid)
