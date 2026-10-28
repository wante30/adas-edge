from competition_io.hud_client import HudClient
from competition_io.hud_simulator import HudSimulator
from competition_io.hud_v1 import Error, Register as R
from tools.competition_console.hud_diagnostics import (
    HudDiagnostics,
    LegacyHudBus,
)
from tools.run_competition_console import dispatch_diagnostic


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def ready_bus():
    bus = HudSimulator()
    bus.advance_frames()
    return bus


def test_gui_diagnostic_requires_enable_then_explicit_command_before_write():
    bus = ready_bus()
    diagnostics = HudDiagnostics(bus, clock=bus.clock)

    assert dispatch_diagnostic(diagnostics, 'fpga_test_left')
    assert bus.writes == []
    assert dispatch_diagnostic(diagnostics, 'fpga_test_enable')
    assert diagnostics.enabled and bus.writes == []

    assert dispatch_diagnostic(diagnostics, 'fpga_test_left')
    assert len(bus.writes) == 1 and bus.writes[0][0] == R.REQUEST
    assert diagnostics.status['phase'] == 'PENDING'

    bus.advance_frames()
    assert diagnostics.poll()['phase'] == 'APPLIED'


def test_initialization_enable_poll_and_close_never_write():
    bus = ready_bus()
    diagnostics = HudDiagnostics(bus)

    assert diagnostics.status["phase"] == "DISABLED"
    assert diagnostics.poll()["phase"] == "DISABLED"
    diagnostics.set_enabled(True)
    assert diagnostics.status["phase"] == "IDLE"
    diagnostics.set_enabled(False)
    diagnostics.close()

    assert bus.writes == []


def test_explicit_left_click_and_off_preserve_unrelated_legacy_fields():
    bus = ready_bus()
    baseline = 0x101 | 0x10 | 0x20 | 0x40 | 0x80
    assert HudClient(bus, clock=bus.clock, sleep=bus.sleep).submit_full_state(
        True, baseline
    ).applied
    initial_writes = len(bus.writes)
    diagnostics = HudDiagnostics(bus, enabled=True, clock=bus.clock)

    pending = diagnostics.request("left")
    assert pending["phase"] == "PENDING"
    assert len(bus.writes) == initial_writes + 1
    assert bus.writes[-1][0] == R.REQUEST
    assert bus.writes[-1][1] & 0x1FFF == (0x1000 | (1 << 9) | (baseline | 2))

    bus.advance_frames()
    applied = diagnostics.poll()
    assert applied["phase"] == "APPLIED" and applied["applied"]
    assert applied["effective_valid"]

    diagnostics.request("off")
    bus.advance_frames()
    off = diagnostics.poll()
    assert off["phase"] == "APPLIED"
    assert bus.read32(R.RK_STATE) & (0x10 | 0x20 | 0x40 | 0x80) == (0x10 | 0x20 | 0x40 | 0x80)
    assert bus.read32(R.RK_STATE) & 0x6 == 0


def test_invalid_legacy_snapshot_blocks_command_before_any_write():
    bus = ready_bus()
    bus.effective = 0x200
    diagnostics = HudDiagnostics(bus, enabled=True, clock=bus.clock)

    status = diagnostics.request("right")

    assert status["phase"] == "BLOCKED"
    assert "reserved bits" in status["reason"]
    assert bus.writes == []


def test_rejected_ack_reports_error_code():
    class RejectingBus(HudSimulator):
        def write32(self, offset, value):
            if offset == R.REQUEST:
                self.writes.append((int(offset), int(value)))
                self._reject(Error.BAD_COMBINATION, value)
                return
            super().write32(offset, value)

    bus = RejectingBus()
    bus.advance_frames()
    diagnostics = HudDiagnostics(bus, enabled=True, clock=bus.clock)

    status = diagnostics.request("left")

    assert status["phase"] == "REJECTED"
    assert status["error"] == Error.BAD_COMBINATION
    assert len(bus.writes) == 1 and bus.writes[0][0] == R.REQUEST


def test_ack_timeout_retains_request_and_poll_resolves_without_resend():
    bus = ready_bus()
    bus.pixel_running = False
    clock = FakeClock()
    diagnostics = HudDiagnostics(bus, enabled=True, timeout_ms=10, clock=clock)

    pending = diagnostics.request("right")
    assert pending["phase"] == "PENDING" and pending["pending"]
    writes_after_request = list(bus.writes)
    clock.now = 1.0
    timed_out = diagnostics.poll()
    assert timed_out["phase"] == "TIMEOUT" and timed_out["pending"]
    assert bus.writes == writes_after_request

    bus.pixel_running = True
    bus.advance_frames()
    applied = diagnostics.poll()
    assert applied["phase"] == "APPLIED" and applied["applied"]
    assert bus.writes == writes_after_request


def test_unknown_and_covered_baselines_do_not_send():
    bus = HudSimulator()
    diagnostics = HudDiagnostics(bus, enabled=True, clock=bus.clock)
    assert diagnostics.request("left")["phase"] == "BLOCKED"
    assert "effective_state_unconfirmed" in diagnostics.status["reason"]
    assert bus.writes == []

    bus = ready_bus()
    bus.dehaze_input = True
    bus.advance_frames()
    diagnostics = HudDiagnostics(bus, enabled=True, clock=bus.clock)
    assert diagnostics.request("right")["phase"] == "BLOCKED"
    assert "dehaze_override_active" in diagnostics.status["reason"]
    assert bus.writes == []


def test_close_does_not_send_automatic_off_or_release():
    bus = ready_bus()
    diagnostics = HudDiagnostics(bus, enabled=True, clock=bus.clock)
    diagnostics.request("left")
    before_close = list(bus.writes)

    diagnostics.close()

    assert bus.writes == before_close
    assert len(bus.writes) == 1
    assert bus.writes[0][0] == R.REQUEST


def test_restricted_legacy_bus_never_writes_v2_or_video_registers():
    raw = ready_bus()
    bus = LegacyHudBus(raw)
    command = (1 << 16) | 0x1000 | 0x101

    for offset in (0x400, 0x638, 0x63C, 0x640, 0x644):
        try:
            bus.write32(offset, command)
        except ValueError:
            pass
        else:
            raise AssertionError(f"legacy bus unexpectedly allowed write to {offset:#x}")
    assert raw.writes == []

    bus.write32(R.REQUEST, command)
    assert raw.writes == [(R.REQUEST, command)]
