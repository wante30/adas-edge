import pytest

from competition_io.hud_client import HudClient, IncoherentSnapshot, OutstandingTransaction, UnsupportedFirmware
from competition_io.hud_simulator import HudSimulator
from competition_io.hud_v1 import Error, Register as R, Request, Status


def client(bus):
    return HudClient(bus, clock=bus.clock, sleep=bus.sleep)


def test_acceptance_is_not_applied_until_real_frame_edge():
    bus = HudSimulator()
    bus.write32(R.REQUEST, Request(1, True, 0x103, 1).encode())
    assert bus.read32(R.STATUS) & Status.BUSY
    assert bus.read32(R.STATUS) & Status.ACCEPTED_VALID
    assert not bus.read32(R.STATUS) & Status.APPLIED_VALID
    bus.advance_frames()
    assert bus.read32(R.APPLIED_SEQ) == 1
    assert bus.read32(R.RK_STATE) == 0x1303
    assert bus.read32(R.EFFECTIVE_STATE) == 0x103


@pytest.mark.parametrize("supports,version,caps", [(False, 0x10000, 63), (True, 0x20000, 63), (True, 0x10000, 31)])
def test_incompatible_firmware_never_receives_a_write(supports, version, caps):
    bus = HudSimulator(supports_hud=supports)
    bus.version, bus.caps = version, caps
    with pytest.raises(UnsupportedFirmware):
        client(bus).submit_full_state(True, 0x101)
    assert bus.writes == [] and bus.video_rejects == 0


def test_pixel_clock_timeout_preserves_pending_and_resolves_after_resume():
    bus = HudSimulator()
    bus.pixel_running = False
    sender = client(bus)
    result = sender.submit_full_state(True, 0x103, 1)
    assert result.phase == "UNKNOWN" and result.accepted and not result.applied
    assert sender.pending and bus.read32(R.STATUS) & Status.BUSY
    with pytest.raises(OutstandingTransaction):
        sender.submit_full_state(True, 0x101)
    assert len(bus.writes) == 1
    bus.pixel_running = True
    bus.advance_frames()
    result = sender.query_pending()
    assert result.applied and sender.pending is None
    assert len(bus.writes) == 1  # query did not execute a second command


def test_dehaze_covers_hide_request_then_restores_latest_committed_target():
    bus = HudSimulator()
    sender = client(bus)
    bus.dehaze_input = True
    result = sender.submit_full_state(True, 0)
    assert result.applied and result.covered and result.effective_state & 0xCF == 0xCF
    assert bus.read32(R.RK_STATE) == 0x1000  # requested hidden state is retained
    sender.submit_full_state(True, 0x101)
    bus.dehaze_input = False
    bus.advance_frames()
    assert bus.read32(R.EFFECTIVE_STATE) == 0x101


def test_dehaze_off_phase_is_not_filled_by_rk_constant_arrow():
    bus = HudSimulator()
    bus.dehaze_input = True
    sender = client(bus)
    sender.submit_full_state(True, 0x103)
    bus.advance_frames(20)
    assert bus.read32(R.EFFECTIVE_STATE) & 0xE == 0
    assert bus.read32(R.EFFECTIVE_STATE) & 0xC1 == 0xC1


def test_twenty_on_twenty_off_and_duplicate_does_not_restart_phase():
    bus = HudSimulator()
    word = Request(1, True, 0x103, 1).encode()
    bus.write32(R.REQUEST, word)
    bus.advance_frames()
    phases = []
    for _ in range(40):
        phases.append(bool(bus.read32(R.EFFECTIVE_STATE) & 2))
        bus.write32(R.REQUEST, word)
        bus.advance_frames()
    assert phases == [True]*20 + [False]*20
    assert bus.rejects == 0


def test_unrelated_field_update_does_not_restart_blink():
    bus = HudSimulator()
    bus.write32(R.REQUEST, Request(1, True, 0x103, 1).encode())
    bus.advance_frames(20)
    bus.write32(R.REQUEST, Request(2, True, 0x113, 1).encode())  # only low beam changed
    bus.advance_frames()
    assert not bus.read32(R.EFFECTIVE_STATE) & 2
    assert bus.read32(R.EFFECTIVE_STATE) & 0x10


def test_hazard_controls_both_arrows_and_restores_independent_targets():
    bus = HudSimulator()
    sender = client(bus)
    sender.submit_full_state(True, 0x109, 4)
    assert bus.effective & 0xE == 0xE
    bus.advance_frames(20)
    assert bus.effective & 0xE == 0
    sender.submit_full_state(True, 0x105)
    assert bus.effective & 0xE == 4


def test_busy_conflict_invalid_writes_leave_pending_and_video_unchanged():
    bus = HudSimulator()
    original = Request(7, True, 0x103, 1)
    bus.write32(R.REQUEST, original.encode())
    for offset, word, kwargs, error in [
        (R.REQUEST, Request(8, True, 0x101).encode(), {}, Error.BUSY),
        (R.REQUEST, Request(7, True, 0x101).encode(), {}, Error.SEQUENCE_CONFLICT),
        (R.REQUEST, original.encode(), {"byte_enable": 3}, Error.BAD_ACCESS),
        (R.REQUEST, original.encode(), {"length": 2}, Error.BAD_ACCESS),
        (R.MAGIC, original.encode(), {}, Error.READ_ONLY_WRITE),
        (0x609, original.encode(), {}, Error.BAD_ACCESS),
    ]:
        bus.write_access(offset, word, **kwargs)
        assert bus.error == error and bus.pending == original and bus.video_rejects == 0
    bus.advance_frames()
    assert bus.applied == original


def test_historical_error_does_not_reject_next_valid_transaction():
    bus = HudSimulator()
    bus.write32(R.REQUEST, 0)
    assert bus.error == Error.ZERO_SEQUENCE
    result = client(bus).submit_full_state(True, 0x101)
    assert result.applied
    assert bus.error == Error.ZERO_SEQUENCE  # sticky historical error


def test_sequence_wrap_and_startup_adopt_existing_pending_transaction():
    bus = HudSimulator()
    bus.write32(R.REQUEST, Request(65535, True, 0x101).encode())
    result = client(bus).submit_full_state(True, 0x111)
    assert result.applied and result.sequence == 1 and bus.effective == 0x111


def test_reset_cancels_pending_without_replaying_old_command():
    bus = HudSimulator()
    bus.pixel_running = False
    sender = client(bus)
    sender.submit_full_state(True, 0x103, 1, timeout_ms=1)
    bus.reset()
    sender.notify_device_reset()
    bus.pixel_running = True
    bus.advance_frames()
    assert bus.effective == 0x101 and bus.applied is None
    assert len(bus.writes) == 1
    result = sender.submit_full_state(True, 0x121)
    assert result.applied and result.sequence == 1


def test_release_restores_default_and_dehaze_still_has_priority():
    bus = HudSimulator()
    sender = client(bus)
    sender.submit_full_state(True, 0x103)
    assert sender.release().applied and bus.effective == 0x101
    bus.dehaze_input = True
    bus.advance_frames()
    assert bus.effective == 0x1CF and not bus.read32(R.STATUS) & Status.RK_OWNER


def test_snapshot_retry_and_read_window_dword_lanes():
    bus = HudSimulator()
    sender = client(bus)
    sender.submit_full_state(True, 0x111)
    for offset in range(0x600, 0x640, 4):
        shift = 8 * (offset & 15)
        assert (bus.read_window128(offset) >> shift) & 0xFFFFFFFF == bus.read32(offset)
    original_read = bus.read32
    reads = 0
    def tear_once(offset):
        nonlocal reads
        reads += 1
        if reads == 4:
            bus.write32(R.MAGIC, 0)  # unrelated state change in the middle of a read
        return original_read(offset)
    bus.read32 = tear_once
    snapshot = sender.snapshot()
    assert snapshot.rejects == 1 and reads > 11


def test_unstable_snapshot_is_bounded_and_never_causes_a_write():
    bus = HudSimulator()
    original_read = bus.read32
    def changing(offset):
        bus._touch()
        return original_read(offset)
    bus.read32 = changing
    with pytest.raises(IncoherentSnapshot):
        client(bus).submit_full_state(True, 0x101)
    assert bus.writes == []


def test_transport_error_after_posted_write_retains_request_for_query():
    bus = HudSimulator()
    sender = client(bus)
    write = bus.write32
    def uncertain_write(offset, value):
        write(offset, value)
        raise OSError("completion lost after device accepted write")
    bus.write32 = uncertain_write
    result = sender.submit_full_state(True, 0x103, 1)
    assert result.phase == "UNKNOWN" and sender.pending
    bus.advance_frames()
    assert sender.query_pending().applied
    assert len(bus.writes) == 1


def test_not_ready_preflight_does_not_write_and_explicit_rejection_is_reported():
    bus = HudSimulator()
    sender = client(bus)
    bus.ready = False
    assert sender.submit_full_state(True, 0x101).reason == "control_not_ready"
    assert bus.writes == []
    bus.ready = True
    write = bus.write32
    def reject_access(offset, value):
        bus.write_access(offset, value, byte_enable=3)
    bus.write32 = reject_access
    result = sender.submit_full_state(True, 0x101)
    assert result.phase == "REJECTED" and result.error == Error.BAD_ACCESS
    assert sender.pending is None
    bus.write32 = write
    assert sender.submit_full_state(True, 0x101).applied


def test_corrupt_applied_record_never_becomes_success():
    bus = HudSimulator()
    sender = client(bus)
    read = bus.read32
    def corrupt(offset):
        value = read(offset)
        return value ^ 0x10 if offset == R.RK_STATE else value
    bus.read32 = corrupt
    result = sender.submit_full_state(True, 0x101)
    assert result.phase == "UNKNOWN" and result.reason == "applied_record_does_not_match_request"
    assert sender.pending is not None
    bus.read32 = read
    assert sender.query_pending().applied
