"""Transport-independent HUD v1 transaction engine.

An actual RK adapter must provide ordered 32-bit MMIO via its existing driver,
serialize all HUD clients per device, and call notify_device_reset on removal,
reset or PCI errors. This module deliberately has no /dev/mem or BAR mapper.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Protocol

from .hud_v1 import CAPS, MAGIC, Register as R, Request, Snapshot, Status


class RegisterBus(Protocol):
    def read32(self, offset: int) -> int: ...
    def write32(self, offset: int, value: int) -> None: ...


class UnsupportedFirmware(RuntimeError):
    pass


class OutstandingTransaction(RuntimeError):
    pass


class IncoherentSnapshot(RuntimeError):
    pass


@dataclass(frozen=True)
class Result:
    phase: str
    sequence: int
    word: int
    accepted: bool = False
    applied: bool = False
    covered: bool = False
    effective_valid: bool = False
    effective_state: int = 0
    error: int = 0
    reason: str = ""


class HudClient:
    def __init__(self, bus: RegisterBus, *, clock=time.monotonic, sleep=time.sleep,
                 lock=None, snapshot_attempts: int = 8):
        self.bus, self.clock, self.sleep = bus, clock, sleep
        # Adapters may share this lock across clients. The driver must also
        # enforce ownership across processes, not just Python threads.
        self.lock = lock if lock is not None else threading.RLock()
        if snapshot_attempts < 1:
            raise ValueError("snapshot_attempts must be positive")
        self.snapshot_attempts = snapshot_attempts
        self.pending: Request | None = None
        self.reject_baseline = 0

    def probe(self) -> None:
        magic = self.bus.read32(R.MAGIC)
        version = self.bus.read32(R.VERSION)
        caps = self.bus.read32(R.CAPS)
        if magic != MAGIC or version >> 16 != 1 or caps & CAPS != CAPS:
            raise UnsupportedFirmware("HUD v1 magic, major version or required capabilities missing")

    def snapshot(self) -> Snapshot:
        registers = (R.STATUS, R.ACCEPTED_SEQ, R.APPLIED_SEQ, R.RK_STATE,
                     R.EFFECTIVE_STATE, R.REJECT_COUNT, R.LAST_ERROR,
                     R.REJECTED_SEQ, R.LAST_ACCEPTED_REQUEST)
        for _ in range(self.snapshot_attempts):
            before = self.bus.read32(R.SNAPSHOT_ID)
            values = [self.bus.read32(reg) for reg in registers]
            after = self.bus.read32(R.SNAPSHOT_ID)
            if any(not isinstance(v, int) or not 0 <= v <= 0xFFFFFFFF for v in [before, after, *values]):
                raise OSError("Invalid MMIO read")
            if before == after:
                status, accepted, applied, rk_state, effective, rejects, error, rejected, last = values
                if (accepted | applied | rejected) & 0xFFFF0000 or rk_state & ~0x1FFF or effective & ~0x1FF:
                    raise OSError("HUD snapshot contains invalid register values")
                return Snapshot(Status(status), accepted, applied, rk_state, effective,
                                rejects, error, rejected, last, after)
        raise IncoherentSnapshot("HUD snapshot changed during every read attempt")

    @staticmethod
    def _deadline(now: float, timeout_ms: float) -> float:
        if not math.isfinite(timeout_ms) or timeout_ms < 0:
            raise ValueError("timeout_ms must be finite and nonnegative")
        return now + timeout_ms / 1000.

    def _unknown(self, reason: str, snap: Snapshot | None = None) -> Result:
        command = self.pending
        accepted = bool(command and snap and snap.status & Status.ACCEPTED_VALID
                        and snap.accepted == command.sequence and snap.last_request == command.encode())
        return Result("UNKNOWN", command.sequence if command else 0,
                      command.encode() if command else 0, accepted=accepted, reason=reason)

    def _resolve(self, snap: Snapshot) -> Result | None:
        command = self.pending
        if command is None:
            return None
        word = command.encode()
        if not snap.status & Status.CONTROL_READY:
            return self._unknown("control_not_ready", snap)
        if (snap.rejects != self.reject_baseline and snap.status & Status.REJECT_VALID
                and snap.rejected == command.sequence):
            self.pending = None
            return Result("REJECTED", command.sequence, word, error=snap.error)
        accepted = bool(snap.status & Status.ACCEPTED_VALID and snap.accepted == command.sequence
                        and snap.last_request == word)
        if snap.status & Status.APPLIED_VALID and snap.applied == command.sequence:
            if not accepted or snap.rk_state != word & 0x1FFF:
                return self._unknown("applied_record_does_not_match_request", snap)
            self.pending = None
            return Result("APPLIED", command.sequence, word, accepted=True, applied=True,
                          covered=bool(snap.status & Status.DEHAZE_OVERRIDE),
                          effective_valid=bool(snap.status & Status.EFFECTIVE_VALID),
                          effective_state=snap.effective)
        return None

    def _wait(self, deadline: float) -> Result:
        last = None
        while True:
            try:
                last = self.snapshot()
                result = self._resolve(last)
                if result is not None:
                    return result
            except (OSError, IncoherentSnapshot) as exc:
                return self._unknown(str(exc))
            remaining = deadline - self.clock()
            if remaining <= 0:
                return self._unknown("timeout_does_not_cancel_request", last)
            self.sleep(min(.001, remaining))

    def submit_full_state(self, owner: bool, state: int, blink: int = 0,
                          timeout_ms: float = 100.) -> Result:
        Request(1, owner, state, blink).encode()  # reject invalid input before IO
        with self.lock:
            deadline = self._deadline(self.clock(), timeout_ms)
            if self.pending is not None:
                raise OutstandingTransaction("Query or resolve the UNKNOWN request before sending a new state")
            self.probe()
            snap = self.snapshot()
            if not snap.status & Status.CONTROL_READY:
                return self._unknown("control_not_ready")
            if snap.status & Status.BUSY:
                if not snap.status & Status.ACCEPTED_VALID:
                    return self._unknown("busy_without_accepted_record")
                self.pending = Request.decode(snap.last_request)
                self.reject_baseline = snap.rejects
                result = self._wait(deadline)
                if not result.applied:
                    return result
                snap = self.snapshot()
            seq = snap.accepted % 65535 + 1 if snap.status & Status.ACCEPTED_VALID else 1
            self.pending = Request(seq, owner, state, blink)
            self.reject_baseline = snap.rejects
            try:
                self.bus.write32(R.REQUEST, self.pending.encode())
                # Ordered read from this device flushes the posted write. It
                # does not prove that the pixel-domain latch has committed.
                if self.bus.read32(R.MAGIC) != MAGIC:
                    return self._unknown("device_disappeared_after_write")
            except OSError as exc:
                return self._unknown(str(exc))
            return self._wait(deadline)

    def query_pending(self) -> Result | None:
        with self.lock:
            if self.pending is None:
                return None
            try:
                self.probe()
                snap = self.snapshot()
                return self._resolve(snap) or self._unknown("still_pending_or_unconfirmed", snap)
            except (OSError, UnsupportedFirmware, IncoherentSnapshot) as exc:
                return self._unknown(str(exc))

    def notify_device_reset(self) -> None:
        """Called by the adapter on actual reset/removal, never on a timeout."""
        with self.lock:
            self.pending = None
            self.reject_baseline = 0

    def release(self, timeout_ms: float = 100.) -> Result:
        return self.submit_full_state(False, 0, timeout_ms=timeout_ms)
