"""HUD v1 wire definitions from the supplied RK/FPGA handoff.

These words control existing icons only. They are not ADAS perception packets.
Offsets are BAR1-relative; this module does not map or access physical memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
import operator


MAGIC = 0x31445548
VERSION = 0x00010000
CAPS = 0x3F


class Register(IntEnum):
    MAGIC = 0x600
    VERSION = 0x604
    REQUEST = 0x608
    STATUS = 0x60C
    ACCEPTED_SEQ = 0x610
    APPLIED_SEQ = 0x614
    RK_STATE = 0x618
    EFFECTIVE_STATE = 0x61C
    REJECT_COUNT = 0x620
    LAST_ERROR = 0x624
    REJECTED_SEQ = 0x628
    SNAPSHOT_ID = 0x62C
    CAPS = 0x630
    LAST_ACCEPTED_REQUEST = 0x634


class Icon(IntFlag):
    VISIBLE = 0x001
    LEFT = 0x002
    RIGHT = 0x004
    HAZARD = 0x008
    LOW_BEAM = 0x010
    HIGH_BEAM = 0x020
    FRONT_FOG = 0x040
    POSITION = 0x080
    ADAS_VISIBLE = 0x100


class Status(IntFlag):
    BUSY = 1
    RK_OWNER = 2
    DEHAZE_OVERRIDE = 4
    ACCEPTED_VALID = 8
    APPLIED_VALID = 16
    REJECT_VALID = 32
    EFFECTIVE_VALID = 64
    CONTROL_READY = 128


class Error(IntEnum):
    NONE = 0
    BUSY = 1
    BAD_ACCESS = 2
    RESERVED_BITS = 3
    BAD_COMBINATION = 4
    ZERO_SEQUENCE = 5
    SEQUENCE_CONFLICT = 6
    READ_ONLY_WRITE = 7
    NOT_READY = 8


class CommandError(ValueError):
    def __init__(self, code: Error):
        self.code = code
        super().__init__(code.name)


@dataclass(frozen=True)
class Request:
    sequence: int
    owner: bool
    state: int
    blink: int = 0  # bit0 left, bit1 right, bit2 hazard

    def encode(self) -> int:
        seq, state, blink = map(operator.index, (self.sequence, self.state, self.blink))
        if not 1 <= seq <= 65535:
            raise CommandError(Error.ZERO_SEQUENCE)
        if self.owner not in (False, True) or not 0 <= state <= 0x1FF or not 0 <= blink <= 7:
            raise CommandError(Error.BAD_COMBINATION)
        if not self.owner and (state or blink):
            raise CommandError(Error.BAD_COMBINATION)
        if any(blink & (1 << i) and not state & mask for i, mask in enumerate((2, 4, 8))):
            raise CommandError(Error.BAD_COMBINATION)
        return (seq << 16) | (int(self.owner) << 12) | (blink << 9) | state

    @classmethod
    def decode(cls, word: int) -> Request:
        word = operator.index(word)
        if not 0 <= word <= 0xFFFFFFFF:
            raise CommandError(Error.BAD_ACCESS)
        if word & 0xE000:
            raise CommandError(Error.RESERVED_BITS)
        request = cls(word >> 16, bool(word & 0x1000), word & 0x1FF, (word >> 9) & 7)
        request.encode()
        return request


@dataclass(frozen=True)
class Snapshot:
    status: Status
    accepted: int
    applied: int
    rk_state: int
    effective: int
    rejects: int
    error: int
    rejected: int
    last_request: int
    snapshot_id: int


VECTORS = (
    (Request(1, True, 0x101), 0x00011101),
    (Request(2, True, 0x103, 1), 0x00021303),
    (Request(3, True, 0x105, 2), 0x00031505),
    (Request(4, True, 0x109, 4), 0x00041909),
    (Request(5, True, 0x1C1), 0x000511C1),
    (Request(6, True, 0x111), 0x00061111),
    (Request(7, True, 0x121), 0x00071121),
    (Request(8, True, 0), 0x00081000),
    (Request(9, False, 0), 0x00090000),
)
