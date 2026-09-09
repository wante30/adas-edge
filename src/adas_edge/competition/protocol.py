from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import ClassVar

from .pixel import PixelFormat
from .policy import AdasAction, LaneState, TrafficLightState, WarningFlags

VERSION = 1
FRAME_MAGIC = b"ADSF"
RESULT_MAGIC = b"ADSR"
FLAG_CRC32 = 1
FRAME_HEADER = struct.Struct("<4sHHIQQHHBBIIII")
RESULT_HEADER = struct.Struct("<4sHHIQQHBBBBIII")
DETECTION = struct.Struct("<HHHHHH")


class ProtocolError(ValueError):
    pass


def _crc(payload: bytes, flags: int) -> int:
    return zlib.crc32(payload) & 0xFFFFFFFF if flags & FLAG_CRC32 else 0


@dataclass(frozen=True)
class DetectionRecord:
    class_id: int
    confidence_q15: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def confidence(self) -> float:
        return self.confidence_q15 / 32767.0

    @classmethod
    def from_detection(cls, detection, width: int, height: int) -> "DetectionRecord | None":
        import math
        x1, y1, x2, y2 = detection.bbox_xyxy
        values = [float(x1), float(y1), float(x2), float(y2)]
        if not all(math.isfinite(value) for value in values):
            return None
        x1, y1, x2, y2 = values
        if x1 > x2 or y1 > y2:
            return None
        x1, x2 = max(0, min(width - 1, int(round(x1)))), max(0, min(width - 1, int(round(x2))))
        y1, y2 = max(0, min(height - 1, int(round(y1)))), max(0, min(height - 1, int(round(y2))))
        return cls(int(detection.class_id) & 0xFFFF, max(0, min(32767, round(float(detection.confidence) * 32767))), x1, y1, x2, y2)

    def pack(self) -> bytes:
        return DETECTION.pack(self.class_id, self.confidence_q15, self.x1, self.y1, self.x2, self.y2)

    @classmethod
    def unpack(cls, data: bytes) -> "DetectionRecord":
        return cls(*DETECTION.unpack(data))


@dataclass(frozen=True)
class FrameDescriptor:
    stream_id: int
    frame_id: int
    timestamp_ns: int
    width: int
    height: int
    pixel_format: PixelFormat | int
    stride: int
    payload: bytes
    flags: int = 0
    magic: ClassVar[bytes] = FRAME_MAGIC

    @property
    def header_len(self) -> int:
        return FRAME_HEADER.size

    def serialize(self) -> bytes:
        payload = bytes(self.payload)
        header = FRAME_HEADER.pack(self.magic, VERSION, FRAME_HEADER.size, self.stream_id, self.frame_id, self.timestamp_ns, self.width, self.height, int(self.pixel_format), 0, self.stride, len(payload), self.flags, _crc(payload, self.flags))
        return header + payload

    @classmethod
    def deserialize(cls, packet: bytes) -> "FrameDescriptor":
        if len(packet) < FRAME_HEADER.size:
            raise ProtocolError("truncated frame header")
        values = FRAME_HEADER.unpack_from(packet)
        magic, version, header_len, stream_id, frame_id, timestamp_ns, width, height, pixel_format, reserved, stride, payload_size, flags, crc32 = values
        if magic != FRAME_MAGIC: raise ProtocolError("bad frame magic")
        if version != VERSION: raise ProtocolError(f"unsupported version: {version}")
        if header_len != FRAME_HEADER.size: raise ProtocolError("invalid frame header length")
        if reserved != 0: raise ProtocolError("reserved field must be zero")
        if len(packet) != header_len + payload_size: raise ProtocolError("wrong frame payload size")
        payload = packet[header_len:]
        if flags & FLAG_CRC32 and crc32 != zlib.crc32(payload) & 0xFFFFFFFF: raise ProtocolError("frame crc mismatch")
        try: PixelFormat(pixel_format)
        except ValueError as exc: raise ProtocolError("unsupported pixel format") from exc
        return cls(stream_id, frame_id, timestamp_ns, width, height, PixelFormat(pixel_format), stride, payload, flags)


@dataclass(frozen=True)
class ResultDescriptor:
    stream_id: int
    frame_id: int
    timestamp_ns: int
    num_objects: int = 0
    traffic_light_state: TrafficLightState | int = TrafficLightState.UNKNOWN
    lane_state: LaneState | int = LaneState.UNKNOWN
    adas_action: AdasAction | int = AdasAction.STOP
    warning_flags: WarningFlags | int = WarningFlags.NONE
    detections: tuple[DetectionRecord, ...] = ()
    flags: int = 0
    magic: ClassVar[bytes] = RESULT_MAGIC

    @property
    def payload(self) -> bytes:
        return b"".join(item.pack() for item in self.detections)

    def serialize(self) -> bytes:
        detections = tuple(self.detections)
        if self.num_objects not in (0, len(detections)):
            raise ProtocolError("num_objects does not match detections")
        count = len(detections)
        payload = self.payload
        header = RESULT_HEADER.pack(self.magic, VERSION, RESULT_HEADER.size, self.stream_id, self.frame_id, self.timestamp_ns, count, int(self.traffic_light_state), int(self.lane_state), int(self.adas_action), 0, int(self.warning_flags), len(payload), _crc(payload, self.flags))
        return header + payload

    @classmethod
    def deserialize(cls, packet: bytes) -> "ResultDescriptor":
        if len(packet) < RESULT_HEADER.size: raise ProtocolError("truncated result header")
        values = RESULT_HEADER.unpack_from(packet)
        magic, version, header_len, stream_id, frame_id, timestamp_ns, count, traffic, lane, action, reserved, warnings, payload_size, crc32 = values
        if magic != RESULT_MAGIC: raise ProtocolError("bad result magic")
        if version != VERSION: raise ProtocolError(f"unsupported version: {version}")
        if header_len != RESULT_HEADER.size: raise ProtocolError("invalid result header length")
        if reserved != 0: raise ProtocolError("reserved field must be zero")
        if payload_size != count * DETECTION.size or len(packet) != header_len + payload_size: raise ProtocolError("wrong result payload size")
        payload = packet[header_len:]
        if crc32 and crc32 != zlib.crc32(payload) & 0xFFFFFFFF: raise ProtocolError("result crc mismatch")
        try:
            traffic, lane, action = TrafficLightState(traffic), LaneState(lane), AdasAction(action)
        except ValueError as exc: raise ProtocolError("invalid result enum") from exc
        detections = tuple(DetectionRecord.unpack(payload[i:i + DETECTION.size]) for i in range(0, len(payload), DETECTION.size))
        return cls(stream_id, frame_id, timestamp_ns, count, traffic, lane, action, WarningFlags(warnings), detections)
