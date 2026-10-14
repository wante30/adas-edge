from dataclasses import replace
import struct
import zlib

import pytest

from competition_ai.scene import SceneInput
from competition_io.adas_status_v1 import (Action, AdasReceiver, AdasStatus, ClockDomain,
                                         Lane, Light, SIZE, Valid, Warning, status_from_scene)


def green(seq=1, **kw):
    return AdasStatus(seq, 0, 12, 0, Light.GREEN, action=Action.FORWARD,
                      valid=Valid.SIGNAL_OBSERVED | Valid.SIGNAL_ASSOCIATED | Valid.DECISION, **kw)


def test_wire_roundtrip_crc_and_q15():
    packet = green(signal_confidence=.8).serialize()
    decoded = AdasStatus.deserialize(packet)
    assert len(packet) == SIZE == 60 and decoded.source_frame_id == 12
    assert decoded.signal_confidence == pytest.approx(.8, abs=1/32767)
    with pytest.raises(ValueError):
        AdasStatus.deserialize(packet[:-1])
    with pytest.raises(ValueError):
        AdasStatus.deserialize(packet[:20] + bytes([packet[20] ^ 1]) + packet[21:])


@pytest.mark.parametrize("offset,value", [(4, 2), (36, 65535), (38, 9), (40, 32), (52, 1)])
def test_valid_crc_does_not_hide_bad_protocol_fields(offset, value):
    packet = bytearray(green().serialize())
    fmt = "<H" if offset in (4, 36, 38) else "<I"
    struct.pack_into(fmt, packet, offset, value)
    struct.pack_into("<I", packet, SIZE-4, zlib.crc32(packet[:-4]))
    with pytest.raises(ValueError):
        AdasStatus.deserialize(packet)


def test_expiry_uses_source_time_duplicate_does_not_refresh_permission():
    now = [0]
    sink = AdasReceiver(ClockDomain.VIDEO_REPLAY, lambda: now[0])
    packet = green(valid_for_ms=100).serialize()
    assert sink.receive(packet).action == Action.FORWARD
    now[0] = 100_000_000
    assert sink.receive(packet).action == Action.UNKNOWN
    assert sink.current().warnings & Warning.STALE


def test_unassociated_green_and_solid_lane_cannot_grant_permission():
    sink = AdasReceiver(ClockDomain.VIDEO_REPLAY, lambda: 0)
    unassociated = replace(green(), valid=Valid.SIGNAL_OBSERVED | Valid.DECISION)
    assert sink.receive(unassociated.serialize()).action == Action.UNKNOWN
    unsafe = replace(green(2), action=Action.LEFT, left_lane=Lane.SOLID,
                     valid=green().valid | Valid.LEFT_LANE)
    assert sink.receive(unsafe.serialize()).action == Action.UNKNOWN
    assert sink.current().warnings & Warning.SOLID_LANE


def test_reordered_data_and_sequence_wrap():
    sink = AdasReceiver(ClockDomain.VIDEO_REPLAY, lambda: 0)
    sink.receive(green(0xFFFFFFFF).serialize())
    red = replace(green(1), light=Light.RED, action=Action.STOP)
    assert sink.receive(red.serialize()).action == Action.STOP
    assert sink.receive(green(0xFFFFFFFE).serialize()).action == Action.STOP
    conflict = replace(red, action=Action.FORWARD)
    assert sink.receive(conflict.serialize()).action == Action.UNKNOWN


def test_clock_domain_or_future_timestamp_never_becomes_fresh():
    sink = AdasReceiver(ClockDomain.FPGA_SOURCE_NS, lambda: 10)
    assert sink.receive(green().serialize()).action == Action.UNKNOWN
    future = replace(green(), clock_domain=ClockDomain.FPGA_SOURCE_NS, source_timestamp_ns=11)
    assert sink.receive(future.serialize()).action == Action.UNKNOWN


def test_other_camera_and_new_sequence_with_older_frame_cannot_replace_current():
    sink = AdasReceiver(ClockDomain.VIDEO_REPLAY, lambda: 100)
    red = replace(green(), light=Light.RED, action=Action.STOP, source_timestamp_ns=100)
    assert sink.receive(red.serialize()).action == Action.STOP
    assert sink.receive(replace(green(2), stream_id=1).serialize()).action == Action.STOP
    assert sink.receive(replace(green(2), source_frame_id=11).serialize()).action == Action.STOP
    assert sink.current().source_frame_id == 12


def test_first_invalid_packet_also_reports_protocol_error():
    sink = AdasReceiver(ClockDomain.VIDEO_REPLAY, lambda: 0)
    assert sink.receive(b"bad").warnings & Warning.PROTOCOL_ERROR


def test_corrupt_packet_does_not_erase_source_ordering_ledger():
    sink = AdasReceiver(ClockDomain.VIDEO_REPLAY, lambda: 100)
    sink.receive(replace(green(), source_timestamp_ns=100).serialize())
    sink.receive(b"bad")
    older = replace(green(2), source_frame_id=11, source_timestamp_ns=99)
    assert sink.receive(older.serialize()).action == Action.UNKNOWN
    assert sink.current().warnings & Warning.PROTOCOL_ERROR
    # An explicit new replay/session is the supported way to reset frame IDs.
    sink.reset_session()
    assert sink.receive(older.serialize()).action == Action.FORWARD


def test_existing_competition_policy_red_dashed_solid_pedestrian_and_unknown():
    def build(scene, **kwargs):
        return status_from_scene(scene, sequence=1, source_frame_id=1, source_timestamp_ns=0, **kwargs)
    red = build(SceneInput('red', vehicle_command='go'), signal_associated=True)
    assert red.action == Action.STOP and red.warnings & Warning.RED_LIGHT
    dashed = build(SceneInput('green', vehicle_command='change_left'), signal_associated=True, left_lane=Lane.DASHED)
    assert dashed.action == Action.LEFT and dashed.valid & Valid.LEFT_LANE
    solid = build(SceneInput('green', vehicle_command='change_right'), signal_associated=True, right_lane=Lane.SOLID)
    assert solid.action == Action.WARNING and solid.warnings & Warning.SOLID_LANE
    pedestrian = build(SceneInput('unknown', pedestrian_hazard=True), signal_associated=False)
    assert pedestrian.action == Action.BRAKE and pedestrian.warnings & Warning.PEDESTRIAN
    unverified = build(SceneInput('green'), signal_associated=False)
    assert unverified.action == Action.UNKNOWN and not unverified.valid & Valid.DECISION
