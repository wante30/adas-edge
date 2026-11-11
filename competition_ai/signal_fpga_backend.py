"""Frame-identified FPGA classifier adapter. No PCIe reads/writes are made.

The future transport supplies decoded responses through the injected lookup;
the tracker supplies the exact native Q4 request identity. This adapter only
validates attributes and cannot grant ego-lane relevance or control authority.
"""
from .multi_signal import Attributes, COLORS, SHAPES
import math


class FPGAAttributeBackend:
    def __init__(self, lookup):
        self.lookup = lookup

    def classify(self, roi, *, request=None):
        if request is None:
            return Attributes(reason='MISSING_BACKEND_REQUEST', backend='fpga_roi_v1')
        response = self.lookup(dict(request))
        try:
            if any(response[key] != request[key] for key in ('frame_id', 'timestamp_ns', 'bbox', 'context', 'source_id')):
                raise ValueError('mismatched FPGA identity')
            color, shape = response['color'], response['shape']
            color_confidence, shape_confidence = response['color_confidence'], response['shape_confidence']
            if color not in COLORS or shape not in SHAPES or not all(
                    math.isfinite(v) and 0 <= v <= 1 for v in (color_confidence, shape_confidence)):
                raise ValueError('invalid FPGA attributes')
            return Attributes(color, shape, color_confidence, shape_confidence,
                              'FPGA_ROI_CLASSIFICATION', 'fpga_roi_v1', {'request': request})
        except (KeyError, TypeError, ValueError):
            return Attributes(reason='FPGA_RESULT_IDENTITY_OR_VALUE_INVALID', backend='fpga_roi_v1')
