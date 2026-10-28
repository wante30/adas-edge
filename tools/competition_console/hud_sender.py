"""Ordered sender for the BAR1 HUD V2 perception mailbox."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import ctypes
import math
import mmap
import operator
import os
from pathlib import Path
import sys
import threading
import time


MAGIC = 0x31445548
MAGIC_OFFSET = 0x600
VERSION_OFFSET = 0x604
CAPS_OFFSET = 0x630
EXT_REQUEST = 0x638
TIME0 = 0x63C
TIME1 = 0x640
EXT_COMMIT = 0x644
EXT_STATUS = 0x648
EXT_APPLIED_SEQ = 0x64C
STATUS_PENDING = 1
STATUS_READY = 2
SEQUENCE_MAX = 0xFFFF


class OutstandingTransaction(RuntimeError):
    pass


class MailboxUnavailable(TimeoutError):
    pass


class UnsupportedFirmware(RuntimeError):
    pass


class Bar1MmapBus:
    """Explicit BAR1 mapping that can write only the four V2 mailbox words."""

    def __init__(self, resource_path, *, map_size=4096):
        if sys.byteorder != "little":
            raise RuntimeError("HUD BAR1 MMIO requires a little-endian host")
        map_size = operator.index(map_size)
        if map_size < EXT_APPLIED_SEQ + 4:
            raise ValueError("BAR1 mapping does not cover the HUD V2 mailbox")
        self.path = Path(resource_path)
        self.mapping = None
        self.fd = None
        self.fd = os.open(str(self.path), os.O_RDWR | getattr(os, "O_SYNC", 0))
        try:
            self.mapping = mmap.mmap(self.fd, map_size, access=mmap.ACCESS_WRITE)
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def _check(self, offset):
        if self.mapping is None:
            raise RuntimeError("HUD BAR1 mapping is closed")
        offset = operator.index(offset)
        if offset % 4 or not MAGIC_OFFSET <= offset <= EXT_APPLIED_SEQ:
            raise ValueError("HUD register must be an aligned DWORD inside the HUD window")
        return offset

    def read32(self, offset):
        offset = self._check(offset)
        return ctypes.c_uint32.from_buffer(self.mapping, offset).value

    def write32(self, offset, value):
        offset = self._check(offset)
        value = operator.index(value)
        if offset not in (EXT_REQUEST, TIME0, TIME1, EXT_COMMIT):
            raise ValueError("Only V2 perception mailbox writes are allowed")
        if not 0 <= value <= 0xFFFFFFFF:
            raise ValueError("MMIO write value must fit one DWORD")
        ctypes.c_uint32.from_buffer(self.mapping, offset).value = value

    def close(self):
        try:
            if self.mapping is not None:
                self.mapping.close()
                self.mapping = None
        finally:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


@dataclass(frozen=True)
class CommitResult:
    phase: str
    sequence: int
    ext_request: int
    time0: int
    time1: int
    committed: bool = False
    reason: str = ""
    applied_seq: int = 0
    mailbox_status: int = 0


def _bcd(value: int) -> int:
    value = operator.index(value)
    if not 0 <= value <= 99:
        raise ValueError("BCD field must be between 0 and 99")
    return (value // 10 << 4) | (value % 10)


def encode_time_words(value: datetime) -> tuple[int, int]:
    """Encode local date/time as the protocol's byte-ordered BCD words."""
    if not isinstance(value, datetime):
        raise TypeError("value must be a datetime")
    time0 = (
        _bcd(value.year % 100)
        | (_bcd(value.year // 100) << 8)
        | (_bcd(value.month) << 16)
        | (_bcd(value.day) << 24)
    )
    time1 = _bcd(value.hour) | (_bcd(value.minute) << 8) | (_bcd(value.second) << 16)
    return time0, time1


def _next_sequence(last_sequence: int) -> int:
    sequence = operator.index(last_sequence)
    if not 0 <= sequence <= SEQUENCE_MAX:
        raise OSError("Invalid HUD extension sequence register")
    return sequence % SEQUENCE_MAX + 1


class HudV2Sender:
    """Send one V2 perception snapshot at a time through an ordered MMIO bus.

    The protocol's applied-sequence register acknowledges the mailbox commit,
    not a separate HDMI pixel-frame boundary. A timed-out commit is retained
    and must be resolved with ``query_pending`` before another write.
    """

    def __init__(self, bus, *, clock=time.monotonic, sleep=time.sleep, poll_interval_ms=1.0):
        self.bus = bus
        self.clock = clock
        self.sleep = sleep
        self.poll_interval = float(poll_interval_ms) / 1000.0
        if not math.isfinite(self.poll_interval) or self.poll_interval <= 0:
            raise ValueError("poll_interval_ms must be finite and positive")
        self.lock = threading.RLock()
        self.pending: CommitResult | None = None
        self.closed = False

    def probe(self):
        if self.closed:
            raise RuntimeError("HUD V2 sender is closed")
        magic = self.bus.read32(MAGIC_OFFSET)
        version = self.bus.read32(VERSION_OFFSET)
        caps = self.bus.read32(CAPS_OFFSET)
        if magic != MAGIC or version >> 16 != 1 or not 0 < caps <= 0xFFFFFFFF:
            raise UnsupportedFirmware("HUD magic, compatible version or V2 capabilities missing")

    @staticmethod
    def _deadline(now: float, timeout_ms: float) -> float:
        timeout_ms = float(timeout_ms)
        if not math.isfinite(timeout_ms) or timeout_ms < 0:
            raise ValueError("timeout_ms must be finite and nonnegative")
        return now + timeout_ms / 1000.0

    def _status(self) -> int:
        status = self.bus.read32(EXT_STATUS)
        if not isinstance(status, int) or not 0 <= status <= 0xFFFFFFFF or status & ~3:
            raise OSError("Invalid HUD extension status register")
        return status

    def _wait_idle(self, deadline: float) -> bool:
        while True:
            status = self._status()
            if not status & STATUS_PENDING and status & STATUS_READY:
                return True
            remaining = deadline - self.clock()
            if remaining <= 0:
                return False
            self.sleep(min(self.poll_interval, remaining))

    def _poll_commit(self, request: CommitResult, deadline: float) -> CommitResult:
        sequence = 0
        while True:
            status = self._status()
            if not status & STATUS_PENDING and status & STATUS_READY:
                sequence = self.bus.read32(EXT_APPLIED_SEQ)
                if not isinstance(sequence, int) or not 0 <= sequence <= SEQUENCE_MAX:
                    raise OSError("Invalid HUD extension applied-sequence register")
                if sequence == request.sequence:
                    self.pending = None
                    return CommitResult(
                        "COMMITTED", request.sequence, request.ext_request,
                        request.time0, request.time1, committed=True,
                        applied_seq=sequence, mailbox_status=status,
                    )
            remaining = deadline - self.clock()
            if remaining <= 0:
                return CommitResult(
                    "UNKNOWN", request.sequence, request.ext_request,
                    request.time0, request.time1, reason="commit_not_confirmed_before_timeout",
                    applied_seq=sequence, mailbox_status=status,
                )
            self.sleep(min(self.poll_interval, remaining))

    def commit(self, ext_request: int, *, time_enable: bool = False,
               timestamp: datetime | None = None,
               timeout_ms: float = 100.0) -> CommitResult:
        ext_request = operator.index(ext_request)
        if not 0 <= ext_request <= 0xFFF:
            raise ValueError("ext_request must fit the defined 12-bit V2 field")
        if any((ext_request >> shift) & 3 == 3 for shift in (1, 6, 8)):
            raise ValueError("Traffic and lane fields must use defined V2 values 0, 1 or 2")
        if not isinstance(time_enable, bool):
            raise TypeError("time_enable must be bool")
        if time_enable:
            if timestamp is None:
                raise ValueError("A timestamp is required when time_enable is true")
            time0, time1 = encode_time_words(timestamp)
            ext_request |= 1 << 11
        else:
            time0, time1 = 0, 0
            ext_request &= ~(1 << 11)
        deadline = self._deadline(self.clock(), timeout_ms)

        with self.lock:
            self.probe()
            if self.pending is not None:
                raise OutstandingTransaction("Resolve the pending HUD V2 commit before sending another")
            if not self._wait_idle(deadline):
                raise MailboxUnavailable("HUD V2 mailbox did not become ready before timeout")

            last_sequence = self.bus.read32(EXT_APPLIED_SEQ)
            sequence = _next_sequence(last_sequence)
            request = CommitResult("PENDING", sequence, ext_request, time0, time1)
            self.pending = request
            try:
                self.bus.write32(EXT_REQUEST, ext_request)
                self.bus.write32(TIME0, time0)
                self.bus.write32(TIME1, time1)
                self.bus.write32(EXT_COMMIT, sequence)
                return self._poll_commit(request, deadline)
            except OSError as exc:
                return self._unknown(request, str(exc))

    @staticmethod
    def _unknown(request, reason):
        return CommitResult(
            "UNKNOWN", request.sequence, request.ext_request,
            request.time0, request.time1, reason=reason,
        )

    def query_pending(self, *, timeout_ms: float = 0.0) -> CommitResult | None:
        deadline = self._deadline(self.clock(), timeout_ms)
        with self.lock:
            if self.closed:
                raise RuntimeError("HUD V2 sender is closed")
            if self.pending is None:
                return None
            try:
                self.probe()
                return self._poll_commit(self.pending, deadline)
            except (OSError, UnsupportedFirmware) as exc:
                return self._unknown(self.pending, str(exc))

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                close = getattr(self.bus, "close", None)
                if close is not None:
                    close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
