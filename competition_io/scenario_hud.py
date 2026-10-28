"""Explicit native-scenario to HUD1 icon projection, with a software-only sink.

HUD1 has no straight/stop/brake, signal colour, lane type or freshness fields.
Those meanings stay in the sidecar; this module never produces legacy ADST
actions or opens a device. LED0/1/2 masks are not HUD1 icon masks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from competition_ai.scenario import Controls, Decision, Direction, Maneuver, Perception
from competition_ai.scenario_policy import decide
from .hud_client import HudClient
from .hud_simulator import HudSimulator
from .hud_v1 import Icon, Register, Request


@dataclass(frozen=True)
class HudProjection:
    owner: bool
    state: int
    blink: int
    sidecar: dict


def project_scenario(controls: Controls, perception: Perception, decision: Decision,
                     *, now_ns: int, source_frame_id: int,
                     source_timestamp_ns: int, valid_for_ms: int = 750) -> HudProjection:
    """Check the source envelope and project current policy to existing icons.

    All timestamps must share the caller's source clock. Invalid/expired source
    or a decision inconsistent with current inputs suppresses directional icons
    and requests the existing warning icon. HUD1 itself cannot expire a command.
    """
    for name, value in (("now_ns", now_ns), ("source_frame_id", source_frame_id),
                        ("source_timestamp_ns", source_timestamp_ns)):
        if type(value) is not int or not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"{name} must be an unsigned 64-bit integer")
    if type(valid_for_ms) is not int or not 1 <= valid_for_ms <= 60000:
        raise ValueError("valid_for_ms must be in 1..60000")
    age_ns = now_ns-source_timestamp_ns
    validity = "valid"
    if age_ns < 0:
        validity = "source_timestamp_in_future"
    elif age_ns >= valid_for_ms*1_000_000:
        validity = "source_expired"
    checked = decide(controls, perception, now_ns)
    if validity == "valid" and decision != checked:
        validity = "decision_does_not_match_current_inputs"

    direction = checked.direction if validity == "valid" else Direction.NONE
    warning = validity != "valid" or bool(checked.alarms) or checked.maneuver == Maneuver.BRAKE
    state = int(Icon.VISIBLE | Icon.ADAS_VISIBLE)
    blink = 0
    if warning:
        state |= int(Icon.HAZARD)
        blink = 4
    elif direction == Direction.LEFT:
        state |= int(Icon.LEFT)
        blink = 1
    elif direction == Direction.RIGHT:
        state |= int(Icon.RIGHT)
        blink = 2
    # There is deliberately no mapping from FORWARD/led_mask=2 to Icon.LEFT=2.
    Request(1, True, state, blink).encode()
    sidecar = {
        "schema": "native_scenario_hud_sidecar_v1",
        "transport": "HUD1_ICONS_ONLY_NO_ADST_PACKET",
        "source_frame_id": source_frame_id,
        "source_timestamp_ns": source_timestamp_ns,
        "now_ns": now_ns,
        "age_ms": age_ns/1_000_000,
        "valid_for_ms": valid_for_ms,
        "validity": validity,
        "source_clock_contract": "CALLER_SUPPLIED_SAME_SOURCE_CLOCK",
        "controls": asdict(controls),
        "perception": asdict(perception),
        "requested_decision": asdict(decision),
        "checked_decision": asdict(checked),
        "semantic_decision_valid": validity == "valid",
        "competition_led_mask": checked.led_mask if validity == "valid" else 0,
        "projected_direction": direction.value,
        "warning_icon": warning,
        "hud1_unrepresented": ["maneuver", "straight_arrow", "stop_brake_state",
                               "traffic_light_colour", "lane_boundary_type",
                               "source_timestamp", "source_frame_id", "ttl"],
        "legacy_adst_generated": False,
    }
    return HudProjection(True, state, blink, sidecar)


class MockScenarioHud:
    """Owned in-memory register model; no transport/port/device can be selected."""

    def __init__(self):
        self.simulator = HudSimulator()
        self.client = HudClient(self.simulator, clock=self.simulator.clock,
                                sleep=self.simulator.sleep)

    def submit(self, projection: HudProjection, *, timeout_ms: float = 100.) -> dict:
        result = self.client.submit_full_state(projection.owner, projection.state,
                                              projection.blink, timeout_ms=timeout_ms)
        snapshot = self.client.snapshot()
        decoded = Request.decode(result.word) if result.word else None
        verified = bool(result.applied and decoded is not None
                        and self.simulator.accepted == decoded
                        and self.simulator.applied == decoded
                        and snapshot.accepted == result.sequence
                        and snapshot.applied == result.sequence
                        and snapshot.rk_state == result.word & 0x1FFF)
        return {
            "evidence": "PC_SOFTWARE_HUD_REGISTER_MODEL_NOT_PHYSICAL_PCIE_OR_FPGA",
            "request_register_bar1_relative": int(Register.REQUEST),
            "request_word_hex": f"0x{result.word:08X}",
            "request_bytes_le_hex": result.word.to_bytes(4, "little").hex(),
            "request_decoded": asdict(decoded) if decoded is not None else None,
            "transaction": asdict(result),
            "receiver_snapshot": asdict(snapshot),
            "receiver_verified": verified,
            "projection": {"owner": projection.owner, "state": projection.state,
                           "blink": projection.blink},
            "semantic_sidecar": projection.sidecar,
            "hardware_writes": 0,
        }
