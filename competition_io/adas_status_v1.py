"""Separate ADAS status wire proposal and source-time freshness checks.

This 60-byte, CRC-protected message has no assigned FPGA BAR address. HUD v1
stays unchanged. A hardware consumer needs a matching clock domain (or an
explicit time conversion) before it can validate source timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum, IntFlag
import math
import struct
import zlib


MAGIC = b"ADST"
VERSION = 1
BODY = struct.Struct("<4sHHIIQQBBBBHHIIII")
SIZE = BODY.size + 4


class Light(IntEnum):
    UNKNOWN = 0
    RED = 1
    YELLOW = 2
    GREEN = 3


class Lane(IntEnum):
    UNKNOWN = 0
    SOLID = 1
    DASHED = 2


class Action(IntEnum):
    UNKNOWN = 0
    STOP = 1
    FORWARD = 2
    LEFT = 3
    RIGHT = 4
    BRAKE = 5
    WARNING = 6


class ClockDomain(IntEnum):
    VIDEO_REPLAY = 1
    HOST_MONOTONIC = 2
    FPGA_SOURCE_NS = 3


class Valid(IntFlag):
    SIGNAL_OBSERVED = 1
    SIGNAL_ASSOCIATED = 2
    LEFT_LANE = 4
    RIGHT_LANE = 8
    DECISION = 16


class Warning(IntFlag):
    RED_LIGHT = 1
    SOLID_LANE = 2
    PEDESTRIAN = 4
    COLLISION = 8
    UNKNOWN_SIGNAL = 16
    UNKNOWN_LANE = 32
    STALE = 64
    OVERSPEED = 128
    UNVERIFIED_ASSOCIATION = 256
    PROTOCOL_ERROR = 512


@dataclass(frozen=True)
class AdasStatus:
    sequence: int
    stream_id: int
    source_frame_id: int
    source_timestamp_ns: int
    light: Light = Light.UNKNOWN
    left_lane: Lane = Lane.UNKNOWN
    right_lane: Lane = Lane.UNKNOWN
    action: Action = Action.UNKNOWN
    signal_confidence: float = 0.
    clock_domain: ClockDomain = ClockDomain.VIDEO_REPLAY
    valid: Valid = Valid(0)
    warnings: Warning = Warning(0)
    valid_for_ms: int = 750

    def serialize(self) -> bytes:
        if (not 1 <= self.sequence <= 0xFFFFFFFF or not 0 <= self.stream_id <= 0xFFFFFFFF
                or not 0 <= self.source_frame_id <= 0xFFFFFFFFFFFFFFFF
                or not 0 <= self.source_timestamp_ns <= 0xFFFFFFFFFFFFFFFF
                or not 1 <= self.valid_for_ms <= 60000
                or not math.isfinite(self.signal_confidence) or not 0 <= self.signal_confidence <= 1):
            raise ValueError("Invalid ADAS identity, lifetime or confidence")
        Light(self.light); Lane(self.left_lane); Lane(self.right_lane); Action(self.action); ClockDomain(self.clock_domain)
        if int(self.valid) & ~31 or int(self.warnings) & ~1023:
            raise ValueError("Reserved ADAS flags set")
        if self.valid & Valid.SIGNAL_ASSOCIATED and not self.valid & Valid.SIGNAL_OBSERVED:
            raise ValueError("Signal association needs a signal observation")
        if self.valid & Valid.SIGNAL_OBSERVED and self.light == Light.UNKNOWN:
            raise ValueError("UNKNOWN light cannot be a valid observation")
        if self.valid & Valid.LEFT_LANE and self.left_lane == Lane.UNKNOWN:
            raise ValueError("UNKNOWN left lane cannot be valid")
        if self.valid & Valid.RIGHT_LANE and self.right_lane == Lane.UNKNOWN:
            raise ValueError("UNKNOWN right lane cannot be valid")
        if self.valid & Valid.DECISION and self.action == Action.UNKNOWN:
            raise ValueError("UNKNOWN decision cannot be valid")
        body = BODY.pack(MAGIC, VERSION, SIZE, self.sequence, self.stream_id,
                         self.source_frame_id, self.source_timestamp_ns,
                         int(self.light), int(self.left_lane), int(self.right_lane), int(self.action),
                         round(self.signal_confidence*32767), int(self.clock_domain),
                         int(self.valid), int(self.warnings), self.valid_for_ms, 0)
        return body + struct.pack("<I", zlib.crc32(body))

    @classmethod
    def deserialize(cls, data: bytes) -> AdasStatus:
        if len(data) != SIZE or zlib.crc32(data[:-4]) != struct.unpack_from("<I", data, SIZE-4)[0]:
            raise ValueError("ADAS length or CRC mismatch")
        magic, version, size, seq, sid, fid, ts, light, left, right, action, confidence, clock, valid, warnings, ttl, reserved = BODY.unpack_from(data)
        if magic != MAGIC or version != VERSION or size != SIZE or reserved or confidence > 32767:
            raise ValueError("ADAS magic/version/size/reserved/confidence mismatch")
        message = cls(seq, sid, fid, ts, Light(light), Lane(left), Lane(right), Action(action),
                      confidence/32767., ClockDomain(clock), Valid(valid), Warning(warnings), ttl)
        message.serialize()  # also enforce field consistency on external data
        return message


@dataclass(frozen=True)
class AdasView:
    action: Action = Action.UNKNOWN
    light: Light = Light.UNKNOWN
    warnings: Warning = Warning(0)
    reason: str = "no_status"
    source_frame_id: int | None = None
    age_ms: float | None = None


class AdasReceiver:
    def __init__(self, clock_domain: ClockDomain, now_ns, *, stream_id: int = 0):
        self.clock_domain = ClockDomain(clock_domain)
        self.now_ns = now_ns
        self.stream_id = stream_id
        self.message: AdasStatus | None = None
        self.last_sequence = 0
        self.last_payload: bytes | None = None
        self.last_source: tuple[int, int] | None = None
        self.failure = "no_status"
        self.has_error = False

    def reset_session(self) -> None:
        self.message, self.last_payload, self.last_sequence = None, None, 0
        self.last_source = None
        self.failure = "session_reset"
        self.has_error = False

    def receive(self, data: bytes) -> AdasView:
        try:
            message = AdasStatus.deserialize(data)
            if message.stream_id != self.stream_id:
                return replace(self.current(), reason="other_stream_ignored")
            if message.clock_domain != self.clock_domain:
                raise ValueError("clock_domain_mismatch")
            if message.source_timestamp_ns > self.now_ns():
                raise ValueError("source_timestamp_in_future")
            if self.last_sequence:
                distance = (message.sequence-self.last_sequence) & 0xFFFFFFFF
                if distance == 0 and data != self.last_payload:
                    raise ValueError("same_sequence_different_payload")
                if distance >= 0x80000000:
                    return self.current()  # reordered older message cannot replace current data
                if distance and self.last_source and (message.source_timestamp_ns < self.last_source[0]
                        or message.source_frame_id < self.last_source[1]):
                    return replace(self.current(), reason="older_source_frame_ignored")
            self.message, self.last_payload, self.last_sequence = message, bytes(data), message.sequence
            self.last_source = (message.source_timestamp_ns, message.source_frame_id)
            self.failure = ""
            self.has_error = False
        except ValueError as exc:
            self.message = None
            self.failure = str(exc)
            self.has_error = True
        return self.current()

    def current(self) -> AdasView:
        m = self.message
        if m is None:
            return AdasView(warnings=Warning.PROTOCOL_ERROR if self.has_error else Warning(0), reason=self.failure)
        age = (self.now_ns()-m.source_timestamp_ns)/1_000_000
        if age < 0 or age >= m.valid_for_ms:
            return AdasView(warnings=m.warnings | Warning.STALE, reason="expired_or_clock_rollback",
                            source_frame_id=m.source_frame_id, age_ms=age)
        action = m.action if m.valid & Valid.DECISION else Action.UNKNOWN
        reason, warnings = "valid", m.warnings
        if action in (Action.FORWARD, Action.LEFT, Action.RIGHT):
            if m.light != Light.GREEN or not m.valid & Valid.SIGNAL_ASSOCIATED:
                action, reason = Action.UNKNOWN, "no_verified_green_permission"
                warnings |= Warning.UNVERIFIED_ASSOCIATION
            elif action == Action.LEFT and (not m.valid & Valid.LEFT_LANE or m.left_lane != Lane.DASHED):
                action, reason = Action.UNKNOWN, "left_lane_change_not_verified_dashed"
                warnings |= Warning.SOLID_LANE if m.left_lane == Lane.SOLID else Warning.UNKNOWN_LANE
            elif action == Action.RIGHT and (not m.valid & Valid.RIGHT_LANE or m.right_lane != Lane.DASHED):
                action, reason = Action.UNKNOWN, "right_lane_change_not_verified_dashed"
                warnings |= Warning.SOLID_LANE if m.right_lane == Lane.SOLID else Warning.UNKNOWN_LANE
        return AdasView(action, m.light if m.valid & Valid.SIGNAL_OBSERVED else Light.UNKNOWN,
                        warnings, reason, m.source_frame_id, age)


def status_from_scene(scene, *, sequence: int, source_frame_id: int, source_timestamp_ns: int,
                      signal_associated: bool, left_lane: Lane = Lane.UNKNOWN,
                      right_lane: Lane = Lane.UNKNOWN, stream_id: int = 0,
                      clock_domain: ClockDomain = ClockDomain.VIDEO_REPLAY) -> AdasStatus:
    """Adapt the existing competition policy without inventing lane semantics."""
    from competition_ai.adas import CompetitionAdas
    light = Light[scene.traffic_light_state.upper()]
    valid = Valid(0)
    if light != Light.UNKNOWN:
        valid |= Valid.SIGNAL_OBSERVED
        if signal_associated:
            valid |= Valid.SIGNAL_ASSOCIATED
    if left_lane != Lane.UNKNOWN:
        valid |= Valid.LEFT_LANE
    if right_lane != Lane.UNKNOWN:
        valid |= Valid.RIGHT_LANE
    if scene.vehicle_command == "change_left":
        scene = replace(scene, lane_type=left_lane.name.lower())
    elif scene.vehicle_command == "change_right":
        scene = replace(scene, lane_type=right_lane.name.lower())
    decision = CompetitionAdas().decide(scene)
    action = Action.FORWARD if decision.action == "GO" else Action[decision.action]
    flags = {"PEDESTRIAN_DANGER": Warning.PEDESTRIAN, "COLLISION_WARNING": Warning.COLLISION,
             "RED_LIGHT": Warning.RED_LIGHT, "YELLOW_LIGHT": Warning.UNKNOWN_SIGNAL,
             "TRAFFIC_LIGHT_STATE_UNKNOWN": Warning.UNKNOWN_SIGNAL,
             "LANE_CHANGE_UNSAFE": Warning.UNKNOWN_LANE, "SOLID_LANE_CHANGE_FORBIDDEN": Warning.SOLID_LANE}
    warnings = Warning(0)
    for flag in decision.warning_flags:
        warnings |= flags[flag]
    if not valid & Valid.SIGNAL_ASSOCIATED and not (scene.pedestrian_hazard or scene.obstacle_hazard):
        action = Action.UNKNOWN
        warnings |= Warning.UNVERIFIED_ASSOCIATION
    else:
        valid |= Valid.DECISION
    return AdasStatus(sequence, stream_id, source_frame_id, source_timestamp_ns, light,
                      left_lane, right_lane, action, clock_domain=clock_domain, valid=valid, warnings=warnings)
