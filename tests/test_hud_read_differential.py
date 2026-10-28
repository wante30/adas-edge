"""Register reads against the pre-optimization implementation across transitions."""
import random

from competition_io.hud_client import HudClient
from competition_io.hud_simulator import HudSimulator
from competition_io.hud_v1 import CAPS, MAGIC, VERSION, Register as R, Request, Status


class LegacyHudSimulator(HudSimulator):
    def read32(self, offset):
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


def assert_register_equivalence(before, after):
    state_before, state_after = vars(before).copy(), vars(after).copy()
    # Include write-only, unknown, unaligned and outside-bank offsets.
    for offset in range(0x5FC, 0x644):
        assert before.read32(offset) == after.read32(offset), hex(offset)
        assert before.read_window128(offset) == after.read_window128(offset), hex(offset)
    assert vars(before) == state_before and vars(after) == state_after
    assert vars(before) == vars(after)


def test_reads_match_original_through_accept_apply_blink_override_reject_reset():
    before, after = LegacyHudSimulator(), HudSimulator()
    rng = random.Random(20261016)
    words = [Request(i, owner, state, blink).encode() for i, owner, state, blink in
             [(1, True, 0x103, 1), (2, True, 0x105, 2), (3, True, 0x109, 4),
              (65535, True, 0x1FF, 7), (1, False, 0, 0), (4, True, 0, 0)]]
    operations = [('write', words[0]), ('write', words[0]), ('write', words[1]),
                  ('frames', 1), ('frames', 20), ('frames', 20), ('cover', True),
                  ('frames', 1), ('frames', 20), ('cover', False), ('frames', 1),
                  ('ready', False), ('write', words[1]), ('reset', None)]
    for _ in range(120):
        operations.append(rng.choice([
            ('write', rng.choice(words)), ('invalid', rng.choice([0, 0xE000, 0xFFFFFFFF])),
            ('bad_access', words[0]), ('read_only', words[0]),
            ('frames', rng.randrange(1, 45)), ('cover', bool(rng.randrange(2))),
            ('ready', bool(rng.randrange(2))), ('pixel', bool(rng.randrange(2))),
            ('reset', None), ('caps', rng.randrange(64)), ('version', rng.randrange(1<<18))]))
    assert_register_equivalence(before, after)
    for op, value in operations:
        for bus in (before, after):
            if op in ('write', 'invalid'): bus.write32(R.REQUEST, value)
            elif op == 'bad_access': bus.write_access(R.REQUEST, value, byte_enable=3)
            elif op == 'read_only': bus.write32(R.VERSION, value)
            elif op == 'frames': bus.advance_frames(value)
            elif op == 'cover': bus.dehaze_input = value
            elif op == 'ready': bus.ready = value
            elif op == 'pixel': bus.pixel_running = value
            elif op == 'reset': bus.reset()
            elif op == 'caps': bus.caps = value
            elif op == 'version': bus.version = value
        assert_register_equivalence(before, after)
    before.supports_hud = after.supports_hud = False
    assert_register_equivalence(before, after)


def test_real_client_transactions_and_all_receipts_match_original_reads():
    before, after = LegacyHudSimulator(), HudSimulator()
    clients = [HudClient(bus, clock=bus.clock, sleep=bus.sleep) for bus in (before, after)]
    for i in range(64):
        owner, state, blink = [(True, 0x103, 1), (True, 0x105, 2), (True, 0x109, 4),
                               (True, 0x101, 0), (False, 0, 0)][i % 5]
        before.dehaze_input = after.dehaze_input = i % 7 < 2
        receipts = [client.submit_full_state(owner, state, blink) for client in clients]
        assert receipts[0] == receipts[1]
        assert_register_equivalence(before, after)
