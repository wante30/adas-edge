from datetime import datetime, timezone

import pytest

from tools.competition_console.hud_sender import (
    EXT_APPLIED_SEQ,
    EXT_COMMIT,
    EXT_REQUEST,
    EXT_STATUS,
    MAGIC,
    MAGIC_OFFSET,
    VERSION_OFFSET,
    CAPS_OFFSET,
    STATUS_PENDING,
    STATUS_READY,
    TIME0,
    TIME1,
    HudV2Sender,
    Bar1MmapBus,
    MailboxUnavailable,
    OutstandingTransaction,
    UnsupportedFirmware,
    encode_time_words,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class FakeHudV2Bus:
    def __init__(self, *, sequence=0, complete_after=2):
        self.sequence = sequence
        self.status = STATUS_READY
        self.complete_after = complete_after
        self.status_reads_until_complete = None
        self.writes = []
        self.closed = False
        self.magic = MAGIC

    def read32(self, offset):
        if offset == MAGIC_OFFSET:
            return self.magic
        if offset == VERSION_OFFSET:
            return 0x10000
        if offset == CAPS_OFFSET:
            return 0xFF
        if offset == EXT_STATUS:
            if self.status_reads_until_complete is not None:
                self.status_reads_until_complete -= 1
                if self.status_reads_until_complete <= 0:
                    self.status = STATUS_READY
                    self.status_reads_until_complete = None
            return self.status
        if offset == EXT_APPLIED_SEQ:
            return self.sequence
        raise AssertionError(f"unexpected read {offset:#x}")

    def write32(self, offset, value):
        self.writes.append((offset, value))
        if offset == EXT_COMMIT:
            self.status = STATUS_PENDING
            if self.complete_after is not None:
                self.status_reads_until_complete = self.complete_after
                self.sequence = value

    def close(self):
        self.closed = True


def test_time_words_match_protocol_bcd_byte_order():
    assert encode_time_words(datetime(2026, 10, 18, 14, 30, 18, tzinfo=timezone.utc)) == (
        0x18102026,
        0x00183014,
    )


def test_commit_waits_for_ready_and_applied_sequence_with_nonzero_wrapping_seq():
    bus = FakeHudV2Bus(sequence=0xFFFF)
    clock = FakeClock()
    sender = HudV2Sender(bus, clock=clock, sleep=clock.sleep)
    result = sender.commit(0xA45, time_enable=True,
                           timestamp=datetime(2026, 10, 18, 14, 30, 18), timeout_ms=10)

    assert result.phase == "COMMITTED" and result.committed
    assert result.sequence == 1
    assert bus.writes == [
        (EXT_REQUEST, 0xA45),
        (TIME0, 0x18102026),
        (TIME1, 0x00183014),
        (EXT_COMMIT, 1),
    ]
    assert sender.pending is None


def test_unknown_commit_is_retained_and_query_resolves_without_resubmitting():
    bus = FakeHudV2Bus(complete_after=None)
    clock = FakeClock()
    sender = HudV2Sender(bus, clock=clock, sleep=clock.sleep)
    result = sender.commit(0x45, timestamp=datetime(2026, 1, 2, 3, 4, 5), timeout_ms=2)

    assert result.phase == "UNKNOWN" and result.sequence == 1
    assert sender.pending is not None
    writes_before_query = list(bus.writes)
    with pytest.raises(OutstandingTransaction):
        sender.commit(0x41, timestamp=datetime(2026, 1, 2, 3, 4, 6))

    bus.status = STATUS_PENDING
    bus.sequence = 1
    bus.status_reads_until_complete = 1
    resolved = sender.query_pending(timeout_ms=2)
    assert resolved is not None and resolved.phase == "COMMITTED"
    assert resolved.sequence == 1 and sender.pending is None
    assert bus.writes == writes_before_query


def test_sender_does_not_write_when_mailbox_never_becomes_ready():
    bus = FakeHudV2Bus()
    bus.status = STATUS_PENDING
    clock = FakeClock()
    sender = HudV2Sender(bus, clock=clock, sleep=clock.sleep)

    with pytest.raises(MailboxUnavailable):
        sender.commit(0x45, timestamp=datetime(2026, 1, 2, 3, 4, 5), timeout_ms=2)
    assert bus.writes == []


def test_time_is_off_and_zeroed_by_default_until_explicitly_enabled():
    bus = FakeHudV2Bus()
    clock = FakeClock()
    sender = HudV2Sender(bus, clock=clock, sleep=clock.sleep)
    result = sender.commit(0xA45 | (1 << 11), timestamp=datetime(2026, 10, 18, 14, 30, 18))

    assert result.committed and not result.ext_request & (1 << 11)
    assert result.time0 == result.time1 == 0
    assert bus.writes == [
        (EXT_REQUEST, 0x245),
        (TIME0, 0),
        (TIME1, 0),
        (EXT_COMMIT, 1),
    ]


def test_time_enable_requires_an_explicit_timestamp():
    bus = FakeHudV2Bus()
    with pytest.raises(ValueError, match="timestamp is required"):
        HudV2Sender(bus).commit(0x45, time_enable=True)
    assert bus.writes == []


@pytest.mark.parametrize("value", [-1, 0x1000, 0xFFFFFFFF, 0x6, 0xC0, 0x300])
def test_sender_rejects_reserved_extension_request_bits(value):
    bus = FakeHudV2Bus()
    sender = HudV2Sender(bus)
    with pytest.raises(ValueError):
        sender.commit(value, timestamp=datetime(2026, 1, 2, 3, 4, 5))
    assert bus.writes == []


@pytest.mark.parametrize("status,sequence", [(STATUS_READY, 0), (0, 1), (STATUS_PENDING | STATUS_READY, 1)])
def test_ready_and_matching_sequence_are_both_required(status, sequence):
    bus = FakeHudV2Bus(complete_after=None)
    clock = FakeClock()
    sender = HudV2Sender(bus, clock=clock, sleep=clock.sleep)
    result = sender.commit(0x45, timeout_ms=0)
    assert result.phase == "UNKNOWN"
    bus.status, bus.sequence = status, sequence

    result = sender.query_pending()
    assert result is not None and result.phase == "UNKNOWN"
    assert not result.committed and sender.pending is not None


def test_probe_rejects_unknown_window_without_writing():
    bus = FakeHudV2Bus()
    bus.magic = 0
    sender = HudV2Sender(bus)
    with pytest.raises(UnsupportedFirmware):
        sender.commit(0x45)
    assert bus.writes == []


def test_ambiguous_io_error_retains_transaction_and_resources_close():
    class FaultBus(FakeHudV2Bus):
        def write32(self, offset, value):
            super().write32(offset, value)
            if offset == EXT_COMMIT:
                raise OSError("posted write status unavailable")

    bus = FaultBus()
    clock = FakeClock()
    with HudV2Sender(bus, clock=clock, sleep=clock.sleep) as sender:
        result = sender.commit(0x45)
        assert result.phase == "UNKNOWN" and not result.committed
        assert sender.pending is not None
        resolved = sender.query_pending(timeout_ms=2)
        assert resolved is not None and resolved.committed
        assert resolved.applied_seq == resolved.sequence == 1

    assert bus.closed and sender.closed
    with pytest.raises(RuntimeError, match="closed"):
        sender.commit(0x45)


def test_mmap_bus_only_writes_v2_mailbox_words_and_closes(tmp_path):
    resource = tmp_path / "resource1"
    resource.write_bytes(bytes(4096))
    with Bar1MmapBus(resource) as bus:
        for offset, value in [(EXT_REQUEST, 0xA45), (TIME0, 0x18102026), (TIME1, 0x00183014), (EXT_COMMIT, 1)]:
            bus.write32(offset, value)
            assert bus.read32(offset) == value
        for offset in [0x400, 0x608, 0x638 + 1, EXT_APPLIED_SEQ]:
            with pytest.raises(ValueError):
                bus.write32(offset, 1)
    assert bus.mapping is None and bus.fd is None
    bus.close()
    with pytest.raises(RuntimeError, match="closed"):
        bus.read32(EXT_STATUS)
