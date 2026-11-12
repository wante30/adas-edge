import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "deployment" / "lubancat" / "qt_gpu_probe"
sys.path.insert(0, str(ROOT))

import pcie_protocol as protocol  # noqa: E402


SPEC = json.loads((ROOT / "rk3566_pcie_protocol_v1.json").read_text(encoding="utf-8"))
REGISTERS = {int(row["offset_hex"], 16): row for row in SPEC["registers"]}


def test_critical_hud_scene_navigation_offsets_match_full_contract():
    expected = {
        protocol.OFF_HUD_MAGIC: "MAGIC",
        protocol.OFF_HUD_VERSION: "VERSION",
        protocol.OFF_HUD_REQUEST: "REQUEST",
        protocol.OFF_HUD_STATUS: "STATUS",
        protocol.OFF_HUD_ACCEPTED: "ACCEPTED_SEQ",
        protocol.OFF_HUD_APPLIED: "APPLIED_SEQ",
        protocol.OFF_HUD_CAPS: "CAPS",
        protocol.OFF_SCENE_SHADOW: "SCENE_SHADOW",
        protocol.OFF_SCENE_TIME0: "TIME0_SHADOW",
        protocol.OFF_SCENE_TIME1: "TIME1_SHADOW",
        protocol.OFF_SCENE_COMMIT: "SCENE_COMMIT",
        protocol.OFF_SCENE_STATUS: "SCENE_CDC_STATUS",
        protocol.OFF_SCENE_APPLIED: "SCENE_APPLIED_SEQ",
        protocol.OFF_NAV_SHADOW: "NAV_SHADOW",
        protocol.OFF_NAV_CONTROL: "NAV_CONTROL",
        protocol.OFF_NAV_STATUS: "NAV_STATUS",
        protocol.OFF_NAV_ACTIVE: "NAV_ACTIVE",
    }

    assert {offset: REGISTERS[offset]["name"] for offset in expected} == expected


def test_scene_bits_and_golden_commit_preserve_time_shadow_semantics():
    expected_fields = [
        ("valid", 0, 1), ("traffic", 1, 2), ("person", 3, 1),
        ("obstacle", 4, 1), ("speed_limit", 5, 1), ("left_lane", 6, 2),
        ("right_lane", 8, 2), ("warning", 10, 1), ("show_time", 11, 1),
    ]
    assert protocol.SCENE_BITS == expected_fields
    assert [
        (field["name"], field["lsb"], field["width_bits"])
        for field in SPEC["scene"]["shadow_fields"]
        if field["name"] != "reserved"
    ] == expected_fields
    assert protocol.GOLDEN["scene"]["writes"] == [
        (protocol.OFF_SCENE_SHADOW, 0x00000985),
        (protocol.OFF_SCENE_TIME0, 0x10242026),
        (protocol.OFF_SCENE_TIME1, 0x00053014),
        (protocol.OFF_SCENE_COMMIT, 0x0000002A),
    ]


def test_pcie_action_write_is_not_enabled_by_this_contract():
    # The attached full document describes an ACTION window, while the
    # checked-in FPGA contract explicitly says PCIe action writes are not
    # supported. Keep this guard so traffic integration cannot start sending
    # unsupported 0x6C8 writes by accident.
    assert SPEC["behavior"]["pcie_action_write_supported"] is False
