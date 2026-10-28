"""Explicit, click-triggered legacy HUD lamp diagnostics, separate from perception."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import ctypes
import math
import mmap
import operator
import os
from pathlib import Path
import sys
import time

from competition_io.hud_client import HudClient, IncoherentSnapshot, UnsupportedFirmware
from competition_io.hud_v1 import CommandError, Register as R, Request, Status


READ_REGISTERS = frozenset(register for register in R if register != R.REQUEST)
REGISTER_MASKS = {
    R.STATUS: 0xFF,
    R.ACCEPTED_SEQ: 0xFFFF,
    R.APPLIED_SEQ: 0xFFFF,
    R.REJECTED_SEQ: 0xFFFF,
    R.RK_STATE: 0x1FFF,
    R.EFFECTIVE_STATE: 0x1FF,
}
COMMANDS = {"left": (2, 1), "right": (4, 2), "off": (0, 0)}


class LegacyHudBus:
    """Restrict an ordered transport to legacy reads and the 0x608 request."""

    def __init__(self, bus):
        self.bus = bus
        self.closed = False

    def read32(self, offset):
        if self.closed:
            raise RuntimeError("Legacy HUD bus is closed")
        offset = operator.index(offset)
        if offset not in READ_REGISTERS:
            raise ValueError("Only legacy HUD status registers may be read")
        value = self.bus.read32(offset)
        if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
            raise OSError("Invalid legacy HUD register read")
        if value & ~REGISTER_MASKS.get(offset, 0xFFFFFFFF):
            raise OSError("Legacy HUD register contains reserved bits")
        if offset == R.LAST_ERROR and value > 8:
            raise OSError("Unknown legacy HUD error code")
        return value

    def write32(self, offset, value):
        if self.closed:
            raise RuntimeError("Legacy HUD bus is closed")
        if operator.index(offset) != R.REQUEST:
            raise ValueError("Only the legacy HUD request at 0x608 may be written")
        Request.decode(value)
        self.bus.write32(int(R.REQUEST), int(value))

    def close(self):
        if not self.closed:
            self.closed = True
            close = getattr(self.bus, "close", None)
            if close is not None:
                close()


class LegacyHudMmapBus:
    """A separate BAR1 mapper whose only writable address is legacy 0x608.

    Construct through the diagnostic controller's lazy factory. Mapping does
    not submit a command. The application must be the sole legacy writer.
    """

    def __init__(self, resource_path):
        if sys.byteorder != "little":
            raise RuntimeError("HUD BAR1 MMIO requires a little-endian host")
        self.path = Path(resource_path)
        if self.path.name != "resource1":
            raise ValueError("Legacy diagnostics require an explicit BAR1 resource1 path")
        self.fd = os.open(str(self.path), os.O_RDWR | getattr(os, "O_SYNC", 0))
        self.mapping = None
        try:
            self.mapping = mmap.mmap(self.fd, 4096, access=mmap.ACCESS_WRITE)
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def read32(self, offset):
        if self.mapping is None:
            raise RuntimeError("Legacy HUD mapping is closed")
        offset = operator.index(offset)
        if offset not in READ_REGISTERS:
            raise ValueError("Only legacy HUD status registers may be read")
        return ctypes.c_uint32.from_buffer(self.mapping, offset).value

    def write32(self, offset, value):
        if self.mapping is None:
            raise RuntimeError("Legacy HUD mapping is closed")
        if operator.index(offset) != R.REQUEST:
            raise ValueError("Only the legacy HUD request at 0x608 may be written")
        Request.decode(value)
        ctypes.c_uint32.from_buffer(self.mapping, int(R.REQUEST)).value = value

    def close(self):
        try:
            if self.mapping is not None:
                self.mapping.close()
                self.mapping = None
        finally:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


@dataclass(frozen=True)
class DiagnosticStatus:
    phase: str = "DISABLED"
    enabled: bool = False
    command: str | None = None
    pending: bool = False
    sequence: int = 0
    word: int = 0
    accepted: bool = False
    applied: bool = False
    covered: bool = False
    effective_valid: bool = False
    effective_state: int = 0
    error: int = 0
    reason: str = "explicit_enable_and_click_required"


class HudDiagnostics:
    """Nonblocking request/poll interface; only request() can write.

    Pass a bus for tests or a lazy bus_factory for the GUI. Enabling, polling,
    initialization and closing never send or release a request. A timeout
    retains the transaction; polling can still resolve its later ACK.
    """

    def __init__(self, bus=None, *, bus_factory=None, enabled=False,
                 timeout_ms=100.0, clock=time.monotonic):
        if bus is not None and bus_factory is not None:
            raise ValueError("Choose bus or bus_factory, not both")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        timeout_ms = float(timeout_ms)
        if not math.isfinite(timeout_ms) or timeout_ms <= 0:
            raise ValueError("timeout_ms must be finite and positive")
        self.enabled = enabled
        self.timeout_s = timeout_ms / 1000.0
        self.clock = clock
        self.bus = None
        self.client = None
        self._provided_bus = bus
        self.bus_factory = bus_factory
        self._deadline = None
        self.closed = False
        self._status = DiagnosticStatus(phase="IDLE" if enabled else "DISABLED", enabled=enabled)

    @property
    def status(self):
        return asdict(replace(self._status, enabled=self.enabled,
                              pending=bool(self.client and self.client.pending)))

    def set_enabled(self, enabled):
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        self.enabled = enabled
        if self.client is None or self.client.pending is None:
            self._status = DiagnosticStatus(phase="IDLE" if enabled else "DISABLED", enabled=enabled)
        return self.status

    def _open(self):
        if self.client is not None:
            return
        raw = self._provided_bus
        if raw is None:
            if self.bus_factory is None:
                raise ValueError("No legacy HUD transport configured")
            raw = self.bus_factory()
        self.bus = LegacyHudBus(raw)
        self.client = HudClient(self.bus, clock=self.clock)

    @staticmethod
    def _baseline(snapshot):
        if not snapshot.status & Status.CONTROL_READY:
            raise ValueError("control_not_ready")
        if snapshot.status & Status.BUSY:
            raise ValueError("legacy_mailbox_busy")
        if not snapshot.status & Status.EFFECTIVE_VALID:
            raise ValueError("effective_state_unconfirmed")
        if snapshot.status & Status.DEHAZE_OVERRIDE:
            raise ValueError("dehaze_override_active")
        has_accepted = bool(snapshot.status & Status.ACCEPTED_VALID)
        has_applied = bool(snapshot.status & Status.APPLIED_VALID)
        if has_accepted or has_applied:
            if not has_accepted or not has_applied or snapshot.accepted != snapshot.applied:
                raise ValueError("legacy_request_not_confirmed_applied")
            request = Request.decode(snapshot.last_request)
            if (request.sequence != snapshot.accepted
                    or request.encode() & 0x1FFF != snapshot.rk_state
                    or bool(snapshot.status & Status.RK_OWNER) != request.owner):
                raise ValueError("inconsistent_legacy_request_snapshot")
            if request.owner:
                return request.state, request.blink
        elif snapshot.accepted or snapshot.applied or snapshot.rk_state or snapshot.last_request:
            raise ValueError("unexpected_legacy_request_without_valid_status")
        if snapshot.effective & 0xE:
            raise ValueError("default_direction_blink_state_unknown")
        return snapshot.effective, 0

    def _record(self, result):
        if result.applied:
            phase = "APPLIED"
        elif result.phase == "REJECTED":
            phase = "REJECTED"
        elif self._deadline is not None and self.clock() >= self._deadline:
            phase = "TIMEOUT"
        else:
            phase = "PENDING"
        self._status = DiagnosticStatus(
            phase=phase, enabled=self.enabled, command=self._status.command,
            sequence=result.sequence, word=result.word, accepted=result.accepted,
            applied=result.applied, covered=result.covered,
            effective_valid=result.effective_valid, effective_state=result.effective_state,
            error=result.error,
            reason="ack_timeout_request_retained" if phase == "TIMEOUT" else result.reason,
        )
        return self.status

    def request(self, command):
        if command not in COMMANDS:
            raise ValueError("Diagnostic command must be left, right or off")
        if self.closed:
            raise RuntimeError("HUD diagnostics are closed")
        if not self.enabled:
            self._status = DiagnosticStatus(command=command)
            return self.status
        if self.client is not None and self.client.pending is not None:
            return self.status
        self._status = DiagnosticStatus(phase="BLOCKED", enabled=True, command=command)
        try:
            self._open()
            with self.client.lock:
                self.client.probe()
                snapshot = self.client.snapshot()
                state, blink = self._baseline(snapshot)
                direction, direction_blink = COMMANDS[command]
                state = (state & ~0x6) | direction
                blink = (blink & ~0x3) | direction_blink
                self._deadline = self.clock() + self.timeout_s
                result = self.client.submit_full_state(True, state, blink, timeout_ms=0.0)
            return self._record(result)
        except (OSError, ValueError, CommandError, IncoherentSnapshot, UnsupportedFirmware) as exc:
            self._status = replace(self._status, phase="BLOCKED", reason=str(exc))
            return self.status

    def poll(self):
        if self.closed or self.client is None or self.client.pending is None:
            return self.status
        result = self.client.query_pending()
        return self._record(result) if result is not None else self.status

    def close(self):
        if not self.closed:
            self.closed = True
            if self.bus is not None:
                self.bus.close()
            self._status = replace(self._status, phase="CLOSED", reason="no_automatic_release_sent")
