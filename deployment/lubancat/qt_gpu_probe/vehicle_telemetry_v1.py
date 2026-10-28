#!/usr/bin/env python3
"""LVT1 read-only FPGA vehicle telemetry; independent of Qt, DMA and HUD.

Authoritative contract: rk3566_pcie_vehicle_telemetry_v1.json (2026-10-28).
Hardware must support PCI_IMAGE_MMIO_READ_CMD for BAR1 offsets 0x690..0x6bc.
This is NOT a substitute for the FPGA-provided vehiclemon.c or driver patch.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import tempfile
from pathlib import Path
from typing import Callable

MAGIC = 0x3154564C
VERSION = 0x00010000
CAPS_REQUIRED = 0x7
IOCTL_READ = 0xC008530B
DEVICE = '/dev/pango_pci_driver'
REG = {
    'magic': 0x690, 'version': 0x694, 'generation': 0x698, 'caps': 0x69C,
    'sample_seq': 0x6A0, 'payload': 0x6A4, 'lane_status': 0x6A8,
    'action_status': 0x6AC, 'behavior': 0x6B0, 'accept_count': 0x6B4,
    'action_hold_ms': 0x6B8, 'hsst_stale_ms': 0x6BC,
}
SNAPSHOT = ('sample_seq', 'payload', 'lane_status', 'action_status', 'behavior', 'accept_count')
TYPES = ('unknown', 'solid', 'dashed', 'unknown')
SOURCES = ('none', 'rk', 'hsst', 'none')
ACTIONS = ('none', 'throttle', 'brake', 'lane_left', 'lane_right', 'unknown', 'unknown', 'unknown')
REASONS = {0: 'none', 1: 'red_light', 2: 'left_solid', 3: 'right_solid',
           6: 'sensor_unavailable', 8: 'brake_advisory'}


class TelemetryError(Exception):
    """Unreadable or incompatible telemetry snapshot."""


class CoherenceError(TelemetryError):
    """Generation changed during every attempted snapshot."""


def bits(word: int, shift: int, width: int = 1) -> int:
    return (word >> shift) & ((1 << width) - 1)


def decode(words: dict[str, int], generation: int) -> dict:
    """Pure decode: never invent geometric lane coordinates or control actions."""
    p, lane, act, beh = (words[k] for k in ('payload', 'lane_status', 'action_status', 'behavior'))
    hsst_fresh = bool(bits(lane, 0) and not bits(lane, 1))
    source = SOURCES[bits(lane, 8, 2)]
    if source == 'hsst' and not hsst_fresh:
        source = 'none'
    raw_counts = (bits(p, 0, 8), bits(p, 8, 8))
    count_valid = (bool(bits(p, 20)), bool(bits(p, 21)))
    type_valid = (bool(bits(p, 22)), bool(bits(p, 23)))
    raw_types = (TYPES[bits(p, 16, 2)], TYPES[bits(p, 18, 2)])
    counts = [raw_counts[i] if hsst_fresh and count_valid[i] else None for i in range(2)]
    hsst_types = [raw_types[i] if hsst_fresh and type_valid[i] and
                  not (count_valid[i] and raw_counts[i] == 0) else 'unknown' for i in range(2)]
    effective = (TYPES[bits(lane, 4, 2)], TYPES[bits(lane, 6, 2)])
    # HSST owns the whole packet; never patch one missing HSST side with RK data.
    side_types = hsst_types if source == 'hsst' else list(effective) if source == 'rk' else ['unknown', 'unknown']
    if source != 'hsst':
        counts = [None, None]  # RK scene supplies type, never counts
    action_active = bool(bits(act, 19))
    decision_active = bool(bits(beh, 14))
    allow = bool(beh & 0x0F)
    deny = bool(beh & 0xF0)
    reason = bits(beh, 8, 6)
    if not action_active or not decision_active:
        decision = 'idle'
    elif reason == 8:
        decision = 'warning'
    elif deny:
        decision = 'deny'
    elif allow:
        decision = 'allow'
    else:
        decision = 'unknown'
    last_action = ACTIONS[bits(act, 16, 3)]
    return {
        'schema': 'pango.rk3566.vehicle_status.v1',
        'generation': generation, 'online': True,
        'hsst_sample_seq': words['sample_seq'],
        'hsst_valid': bool(bits(lane, 0)), 'hsst_stale': bool(bits(lane, 1)),
        'hsst_fresh': hsst_fresh, 'hsst_owns_both_sides': bool(bits(lane, 2)),
        'rk_scene_fresh': bool(bits(lane, 3)), 'lane_source': source,
        'left_line_count': counts[0], 'right_line_count': counts[1],
        'left_nearest_type': side_types[0], 'right_nearest_type': side_types[1],
        'current_action': last_action if action_active else 'none',
        'last_action': last_action, 'last_sequence': bits(act, 0, 16),
        'active': action_active, 'timebase_ready': bool(bits(act, 20)),
        'accept_count': words['accept_count'],
        'decision': decision, 'decision_action_active': decision_active,
        'reason_code': reason, 'reason': REASONS.get(reason, 'unknown'),
        'sensor_stale': bool(bits(beh, 15)),
        'throttle_allow': bool(bits(beh, 0)), 'brake_allow': bool(bits(beh, 1)),
        'lane_left_allow': bool(bits(beh, 2)), 'lane_right_allow': bool(bits(beh, 3)),
        'throttle_deny': bool(bits(beh, 4)), 'brake_deny': bool(bits(beh, 5)),
        'lane_left_deny': bool(bits(beh, 6)), 'lane_right_deny': bool(bits(beh, 7)),
        'action_sequence_low8': bits(beh, 16, 8),
        'rk_scene_sequence_low8': bits(beh, 24, 8),
        'action_hold_ms': words['action_hold_ms'], 'hsst_stale_ms': words['hsst_stale_ms'],
    }


def read_snapshot(read32: Callable[[int], int], max_attempts: int = 5) -> dict:
    def read(name: str) -> int:
        v = read32(REG[name])
        if not isinstance(v, int) or not 0 <= v <= 0xFFFFFFFF:
            raise TelemetryError('invalid DWORD read for ' + name)
        return v

    def check_identity() -> None:
        magic, version, caps = read('magic'), read('version'), read('caps')
        if magic != MAGIC or version != VERSION or (caps & CAPS_REQUIRED) != CAPS_REQUIRED:
            raise TelemetryError('LVT1 identity/version/caps mismatch or link down')

    check_identity()
    for _ in range(max_attempts):
        g0 = read('generation')
        values = {name: read(name) for name in SNAPSHOT}
        values['action_hold_ms'] = read('action_hold_ms')
        values['hsst_stale_ms'] = read('hsst_stale_ms')
        g1 = read('generation')
        if g0 == g1:
            check_identity()
            return decode(values, g1)
    raise CoherenceError('generation changed during all LVT1 snapshot attempts')


class IoctlReader:
    def __init__(self, path: str):
        self.fd = os.open(path, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0))

    def __call__(self, offset: int) -> int:
        import fcntl  # Linux-only; decoder/tests can run on Windows
        request = bytearray(struct.pack('<II', offset, 0))
        fcntl.ioctl(self.fd, IOCTL_READ, request, True)
        return struct.unpack('<II', request)[1]

    def close(self) -> None:
        os.close(self.fd)


def write_status_file(path: Path, status: dict) -> None:
    """Atomic, complete snapshot for a future read-only Qt consumer."""
    target = Path(path)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=str(target.parent),
                                         prefix='.lvt1-', delete=False) as output:
            staged = output.name
            json.dump(status, output, ensure_ascii=False, separators=(',', ':'))
            output.write('\n')
        os.replace(staged, target)
    finally:
        if staged and os.path.exists(staged):
            os.unlink(staged)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Read-only LVT1 BAR1 vehicle telemetry')
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--output-json', type=Path, help='optional atomic latest status file')
    parser.add_argument('--interval-ms', type=int, default=25)
    args = parser.parse_args(argv)
    if not 20 <= args.interval_ms <= 50:
        parser.error('--interval-ms must be 20..50')
    try:
        reader = IoctlReader(args.device)
        try:
            while True:
                start = time.monotonic()
                try:
                    snapshot = read_snapshot(reader)
                except CoherenceError as exc:
                    print(str(exc), file=sys.stderr)
                    if args.once:
                        return 4
                    time.sleep(max(0, args.interval_ms / 1000 - (time.monotonic() - start)))
                    continue
                if args.output_json:
                    write_status_file(args.output_json, snapshot)
                print(json.dumps(snapshot, ensure_ascii=False, separators=(',', ':')), flush=True)
                if args.once:
                    return 0
                time.sleep(max(0, args.interval_ms / 1000 - (time.monotonic() - start)))
        finally:
            reader.close()
    except (OSError, TelemetryError, ValueError, BrokenPipeError) as exc:
        print('LVT1 read failed: ' + str(exc), file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
