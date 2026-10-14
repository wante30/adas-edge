"""Executable PC reference model of the planned HUD registers.

This is neither an RTL simulator nor a PCIe/CDC timing model. The discrete
frame edge stands for a successful pixel-domain commit; clock stops, resets,
read tearing, access rejection and display priority can be exercised locally.
"""
from __future__ import annotations

from .hud_v1 import CAPS, MAGIC, VERSION, CommandError, Error, Register as R, Request, Status


class HudSimulator:
    def __init__(self, fps: int = 60, *, supports_hud: bool = True):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.period_ns = round(1_000_000_000 / fps)
        self.now_ns = 0
        self.next_edge_ns = self.period_ns
        self.pixel_running = True
        self.supports_hud = supports_hud
        self.dehaze_input = False
        self.version, self.caps = VERSION, CAPS
        self.writes: list[tuple[int, int]] = []
        self.video_rejects = 0
        self.reset()

    def clock(self) -> float:
        return self.now_ns / 1_000_000_000

    def sleep(self, seconds: float) -> None:
        target = self.now_ns + max(1, round(seconds * 1_000_000_000))
        while self.next_edge_ns <= target:
            self.now_ns = self.next_edge_ns
            if self.pixel_running:
                self.frame_edge()
            self.next_edge_ns += self.period_ns
        self.now_ns = target

    def advance_frames(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be nonnegative")
        for _ in range(count):
            self.sleep((self.next_edge_ns - self.now_ns) / 1_000_000_000)

    def reset(self) -> None:
        self.pending: Request | None = None
        self.accepted: Request | None = None
        self.applied: Request | None = None
        self.ready = True
        self.effective = 0
        self.effective_valid = False
        self.covered = False
        self.rejects = 0
        self.error = Error.NONE
        self.rejected = 0
        self.reject_valid = False
        self.snapshot_id = 0
        self.phases = [0, 0, 0]
        self.dehaze_phase = 0

    def _touch(self) -> None:
        self.snapshot_id = (self.snapshot_id + 1) & 0xFFFFFFFF

    def _reject(self, code: Error, value: int | None) -> None:
        self.rejects = (self.rejects + 1) & 0xFFFFFFFF
        self.error = code
        self.reject_valid = value is not None
        self.rejected = (value >> 16) & 0xFFFF if value is not None else 0
        self._touch()

    def write32(self, offset: int, value: int) -> None:
        self.write_access(offset, value)

    def write_access(self, offset: int, value: int, *, length: int = 1, byte_enable: int = 15) -> None:
        self.writes.append((int(offset), int(value)))
        if not self.supports_hud:
            self.video_rejects += 1  # old video firmware has no HUD bank
            return
        if not 0x600 <= offset < 0x640:
            raise ValueError("This reference model only implements the HUD bank")
        if offset % 4 or length != 1 or byte_enable != 15 or not 0 <= value <= 0xFFFFFFFF:
            self._reject(Error.BAD_ACCESS, value if length == 1 and 0 <= value <= 0xFFFFFFFF else None)
            return
        if offset != R.REQUEST:
            self._reject(Error.READ_ONLY_WRITE, value)
            return
        try:
            command = Request.decode(value)
        except CommandError as exc:
            self._reject(exc.code, value)
            return
        if self.accepted and command.sequence == self.accepted.sequence:
            if command != self.accepted:
                self._reject(Error.SEQUENCE_CONFLICT, value)
            return  # exact duplicate neither restarts phase nor overwrites pending
        if not self.ready:
            self._reject(Error.NOT_READY, value)
        elif self.pending is not None:
            self._reject(Error.BUSY, value)
        else:
            self.accepted = self.pending = command
            self._touch()

    def read32(self, offset: int) -> int:
        if not self.supports_hud:
            return 0
        status = (Status.BUSY if self.pending else 0) | (Status.RK_OWNER if self.applied and self.applied.owner else 0)
        status |= Status.DEHAZE_OVERRIDE if self.covered else 0
        status |= Status.ACCEPTED_VALID if self.accepted else 0
        status |= Status.APPLIED_VALID if self.applied else 0
        status |= Status.REJECT_VALID if self.reject_valid else 0
        status |= Status.EFFECTIVE_VALID if self.effective_valid else 0
        status |= Status.CONTROL_READY if self.ready else 0
        registers = {
            R.MAGIC: MAGIC, R.VERSION: self.version, R.CAPS: self.caps, R.STATUS: int(status),
            R.ACCEPTED_SEQ: self.accepted.sequence if self.accepted else 0,
            R.APPLIED_SEQ: self.applied.sequence if self.applied else 0,
            R.RK_STATE: self.applied.encode() & 0x1FFF if self.applied else 0,
            R.EFFECTIVE_STATE: self.effective, R.REJECT_COUNT: self.rejects, R.LAST_ERROR: int(self.error),
            R.REJECTED_SEQ: self.rejected, R.SNAPSHOT_ID: self.snapshot_id,
            R.LAST_ACCEPTED_REQUEST: self.accepted.encode() if self.accepted else 0,
        }
        return registers.get(offset, 0)

    def read_window128(self, byte_offset: int) -> int:
        base = byte_offset & ~15
        return sum(self.read32(base + 4*i) << (32*i) for i in range(4))

    @staticmethod
    def _mode(command: Request | None, index: int) -> tuple[bool, bool]:
        return (bool(command and command.owner and command.state & (2 << index)),
                bool(command and command.owner and command.blink & (1 << index)))

    def frame_edge(self) -> None:
        old = (self.effective, self.effective_valid, self.covered)
        changed = [False, False, False]
        if self.pending is not None:
            new = self.pending
            for i in range(3):
                if self._mode(new, i) != self._mode(self.applied, i):
                    self.phases[i] = 0
                    changed[i] = True
            self.applied, self.pending = new, None
            self._touch()
        covered = self.dehaze_input
        if self.covered and not covered:
            self.phases = [0, 0, 0]
            changed = [True, True, True]
        elif not covered:
            for i in range(3):
                if not changed[i] and self.effective_valid:
                    self.phases[i] = (self.phases[i] + 1) % 40
        if covered:
            self.dehaze_phase = (self.dehaze_phase + 1) % 40 if self.covered else 0
        command = self.applied
        state = command.state if command and command.owner else 0x101
        blink = command.blink if command and command.owner else 0
        if state & 8:
            state &= ~0xE
            if not blink & 4 or self.phases[2] < 20:
                state |= 0xE
        else:
            for i in range(2):
                if blink & (1 << i) and self.phases[i] >= 20:
                    state &= ~(2 << i)
        if covered:
            # Selection replaces all six covered bits, including OFF phases.
            state = (state & ~0xCF) | 0xC1 | (0xE if self.dehaze_phase < 20 else 0)
        self.effective, self.effective_valid, self.covered = state, True, covered
        if old != (self.effective, self.effective_valid, self.covered):
            self._touch()
