#!/usr/bin/env python3
"""Pure protocol encoding/decoding for RK3566 PCIe v1.0.1.

This module contains no MMIO.  It is shared by the service, simulator, tests
and the browser UI.  All values are integer DWORDs; JSON is only a transport.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any

VERSION = "1.0.1"
MAGIC_HUD = 0x31445548
MAGIC_ISP = 0x31435049
HUD_CAPS = 0x000001FF
NAV_CAP = 0x00000100

OFF_HUD_MAGIC = 0x600
OFF_HUD_VERSION = 0x604
OFF_HUD_CAPS = 0x630
OFF_HUD_REQUEST = 0x608  # protocol v1.0.1: 0x608 is REQUEST, caps are 0x630
OFF_HUD_STATUS = 0x60C
OFF_HUD_ACCEPTED = 0x610
OFF_HUD_APPLIED = 0x614
OFF_HUD_EFFECTIVE = 0x61C
OFF_HUD_REJECT = 0x620
OFF_HUD_ERROR = 0x624
OFF_HUD_LAST_REQUEST = 0x634
OFF_SCENE_SHADOW = 0x638
OFF_SCENE_TIME0 = 0x63C
OFF_SCENE_TIME1 = 0x640
OFF_SCENE_COMMIT = 0x644
OFF_SCENE_STATUS = 0x648
OFF_SCENE_APPLIED = 0x64C
OFF_DIAG = 0x658
OFF_NAV_SHADOW = 0x670
OFF_NAV_CONTROL = 0x674
OFF_NAV_STATUS = 0x678
OFF_NAV_ACTIVE = 0x67C
OFF_HSST_SEQ = 0x680
OFF_HSST_PAYLOAD = 0x684
OFF_HSST_STATUS = 0x688
OFF_HSST_PROTOCOL = 0x68C
OFF_ISP_MAGIC = 0x700
OFF_ISP_VERSION = 0x704
OFF_ISP_CAPS = 0x708
OFF_ISP_GLOBAL = 0x70C
OFF_ISP_TILE = 0x710
OFF_ISP_COMMIT = 0x750
OFF_ISP_STATUS = 0x754
OFF_ISP_RESULT = 0x758
OFF_ISP_CLEAR = 0x75C
OFF_ALGO_LO = 0x760
OFF_ALGO_HI = 0x764
OFF_ALGO_COMMIT = 0x768
OFF_ALGO_STATUS = 0x76C
OFF_ALGO_RESULT = 0x770
OFF_ALGO_CLEAR = 0x774
OFF_ISP_SNAP_INDEX = 0x780
OFF_ISP_SNAP_DATA = 0x784
OFF_ALGO_SNAP_LO = 0x788
OFF_ALGO_SNAP_HI = 0x78C

HUD_BITS = [
    ("hud_enable", 0, 1), ("left", 1, 1), ("right", 2, 1), ("hazard", 3, 1),
    ("low_beam", 4, 1), ("high_beam", 5, 1), ("front_fog", 6, 1),
    ("position", 7, 1), ("adas", 8, 1), ("left_blink", 9, 1),
    ("right_blink", 10, 1), ("hazard_blink", 11, 1), ("takeover", 12, 1),
]
EFFECTIVE_BITS = HUD_BITS[:9]

SCENE_BITS = [
    ("valid", 0, 1), ("traffic", 1, 2), ("person", 3, 1),
    ("obstacle", 4, 1), ("speed_limit", 5, 1), ("left_lane", 6, 2),
    ("right_lane", 8, 2), ("warning", 10, 1), ("show_time", 11, 1),
]

ISP_FIELDS = [
    ("gain_enable", 120, 1, 0, 1, 0, False),
    ("denoise_enable", 121, 1, 0, 1, 0, False),
    ("median_enable", 122, 1, 0, 1, 0, False),
    ("sharpen_enable", 123, 1, 0, 1, 0, False),
    ("tone_enable", 124, 1, 0, 1, 0, False),
    ("saturation_enable", 125, 1, 0, 1, 0, False),
    ("gain_r", 110, 10, 0, 1023, 256, False),
    ("gain_g", 100, 10, 0, 1023, 256, False),
    ("gain_b", 90, 10, 0, 1023, 256, False),
    ("offset_r", 82, 8, -128, 127, 0, True),
    ("offset_g", 74, 8, -128, 127, 0, True),
    ("offset_b", 66, 8, -128, 127, 0, True),
    ("denoise_mode", 64, 2, 0, 1, 1, False),
    ("denoise_strength", 61, 3, 0, 4, 1, False),
    ("denoise_threshold", 53, 8, 0, 255, 12, False),
    ("median_threshold", 45, 8, 0, 255, 32, False),
    ("sharpen_strength", 42, 3, 0, 4, 1, False),
    ("sharpen_noise", 34, 8, 0, 255, 4, False),
    ("sharpen_limit", 26, 8, 0, 255, 16, False),
    ("tone_profile", 23, 3, 0, 3, 0, False),
    ("saturation_level", 20, 3, 0, 4, 0, False),
]
ISP_RESET_DW = [0x40000000, 0x21840410, 0x00000001, 0x00401004]
ISP_GLOBAL_RESET = 0x1E

ALGO_FIELDS = [
    ("stage_mask", 56, 8, 0, 15, 0),
    ("tile_mask", 48, 8, 0, 15, 15),
    ("w", 40, 8, 0, 255, 230),
    ("t_min", 32, 8, 1, 255, 25),
    ("atmosphere_a", 24, 8, 1, 250, 220),
    ("flags", 16, 8, 0, 1, 1),
    ("threshold", 0, 16, 0, 1443, 128),
]
ALGO_RESET = 0x000FE619DC010080
ALGO_STAGE = {"dehaze": 0, "gaussian": 1, "median": 2, "sobel": 3}


def u32(v: Any) -> int:
    return int(v) & 0xFFFFFFFF


def u64(v: Any) -> int:
    return int(v) & 0xFFFFFFFFFFFFFFFF


def pack_field(value: int, lsb: int, width: int) -> int:
    mask = (1 << width) - 1
    return (int(value) & mask) << lsb


def extract(value: int, lsb: int, width: int) -> int:
    return (int(value) >> lsb) & ((1 << width) - 1)


def signed_extract(value: int, lsb: int, width: int) -> int:
    v = extract(value, lsb, width)
    return v - (1 << width) if v & (1 << (width - 1)) else v


def bit(value: int, n: int) -> bool:
    return bool((int(value) >> n) & 1)


def hud_request(seq: int, fields: dict[str, Any], takeover: bool = True) -> int:
    seq = int(seq)
    if not 1 <= seq <= 0xFFFF:
        raise ValueError("HUD sequence must be 1..65535")
    out = seq << 16
    for name, lsb, _ in HUD_BITS:
        if fields.get(name):
            out |= 1 << lsb
    if takeover:
        out |= 1 << 12
    if fields.get("left_blink") and not fields.get("left"):
        raise ValueError("left_blink requires left")
    if fields.get("right_blink") and not fields.get("right"):
        raise ValueError("right_blink requires right")
    if fields.get("hazard_blink") and not fields.get("hazard"):
        raise ValueError("hazard_blink requires hazard")
    if fields.get("left") and fields.get("right"):
        raise ValueError("left and right cannot both be set")
    return u32(out)


def hud_decode_effective(v: int) -> dict[str, bool]:
    return {name: bit(v, lsb) for name, lsb, _ in EFFECTIVE_BITS}


def hud_decode_request(v: int) -> dict[str, Any]:
    out: dict[str, Any] = {name: bit(v, lsb) for name, lsb, _ in HUD_BITS}
    out["sequence"] = extract(v, 16, 16)
    out["reserved"] = extract(v, 13, 3)
    return out


def scene_shadow(fields: dict[str, Any]) -> int:
    out = 0
    for name, lsb, width in SCENE_BITS:
        value = int(bool(fields.get(name))) if width == 1 else int(fields.get(name, 0))
        if not 0 <= value < (1 << width):
            raise ValueError(f"SCENE {name} out of range")
        out |= pack_field(value, lsb, width)
    return u32(out)


def scene_decode(v: int) -> dict[str, Any]:
    return {name: (bit(v, lsb) if width == 1 else extract(v, lsb, width))
            for name, lsb, width in SCENE_BITS}


def bcd_time0(dt: _dt.datetime) -> int:
    return int(f"{dt.month:02d}{dt.day:02d}{dt.year:04d}", 16)


def bcd_time1(dt: _dt.datetime) -> int:
    return int(f"00{dt.second:02d}{dt.minute:02d}{dt.hour:02d}", 16)


def bcd_validate(time0: int, time1: int) -> tuple[bool, str]:
    try:
        s0 = f"{u32(time0):08x}"
        s1 = f"{u32(time1):08x}"
        month, day, year = int(s0[0:2]), int(s0[2:4]), int(s0[4:8])
        hour, minute, second = int(s1[2:4]), int(s1[4:6]), int(s1[6:8])
        _dt.datetime(year, month, day, hour, minute, second)
        return True, ""
    except Exception as exc:
        return False, f"invalid BCD date/time: {exc}"


def nav_shadow(enabled: bool, direction: str, distance_m: int) -> int:
    dirs = {"straight": 0, "left": 1, "right": 2}
    if direction not in dirs:
        raise ValueError("NAV direction must be straight/left/right")
    distance_m = int(distance_m)
    if not 0 <= distance_m <= 65535:
        raise ValueError("NAV distance must be 0..65535")
    out = (1 if enabled else 0) | (dirs[direction] << 1) | (distance_m << 16)
    return u32(out)


def nav_decode(v: int) -> dict[str, Any]:
    dirs = {0: "straight", 1: "left", 2: "right", 3: "invalid"}
    return {"enabled": bit(v, 0), "direction": dirs[extract(v, 1, 2)],
            "distance_m": extract(v, 16, 16)}


def nav_control(seq: int, apply: bool = True) -> int:
    if not 0 <= seq <= 0xFFFF:
        raise ValueError("NAV sequence must be 0..65535")
    return u32((seq << 16) | (1 if apply else 2))


def isp_tile_encode(fields: dict[str, Any]) -> int:
    out = 0
    for name, lsb, width, lo, hi, default, signed in ISP_FIELDS:
        value = int(fields.get(name, default))
        if value < lo or value > hi:
            raise ValueError(f"ISP {name} out of range [{lo},{hi}]")
        if signed:
            value &= (1 << width) - 1
        out |= pack_field(value, lsb, width)
    return int.from_bytes(out.to_bytes(16, "little"), "little")


def isp_tile_decode(value: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, lsb, width, lo, hi, default, signed in ISP_FIELDS:
        out[name] = signed_extract(value, lsb, width) if signed else extract(value, lsb, width)
    return out


def isp_tile_dwords(value: int) -> list[int]:
    return [extract(value, i * 32, 32) for i in range(4)]


def isp_words_to_value(words: list[int]) -> int:
    if len(words) != 4:
        raise ValueError("ISP tile needs four DWORDs")
    return sum(u32(w) << (32 * i) for i, w in enumerate(words))


def isp_reset_fields() -> dict[str, Any]:
    return {name: default for name, _lsb, _w, _lo, _hi, default, _s in ISP_FIELDS}


def algo_encode(fields: dict[str, Any]) -> int:
    out = 0
    for name, lsb, width, lo, hi, default in ALGO_FIELDS:
        value = int(fields.get(name, default))
        if value < lo or value > hi:
            raise ValueError(f"ALGO {name} out of range [{lo},{hi}]")
        out |= pack_field(value, lsb, width)
    return u64(out)


def algo_decode(value: int) -> dict[str, Any]:
    return {name: extract(value, lsb, width)
            for name, lsb, width, _lo, _hi, _default in ALGO_FIELDS}


def algo_with_stage(value: int, stage: str, enable: bool) -> int:
    if stage not in ALGO_STAGE:
        raise ValueError(f"unknown stage {stage}")
    value = u64(value)
    mask = 1 << (56 + ALGO_STAGE[stage])
    return (value | mask) if enable else (value & ~mask)


def algo_stages(value: int) -> dict[str, bool]:
    return {name: bit(value, 56 + n) for name, n in ALGO_STAGE.items()}


def behavior_decode(v: int) -> dict[str, Any]:
    names = ["throttle_allow", "brake_allow", "left_allow", "right_allow",
             "throttle_forbid", "brake_forbid", "left_forbid", "right_forbid"]
    out: dict[str, Any] = {name: bit(v, i) for i, name in enumerate(names)}
    out["reason"] = extract(v, 8, 6)
    out["action_active"] = bit(v, 14)
    return out


def behavior_encode(sample: dict[str, Any]) -> int:
    out = 0
    for i, name in enumerate(["throttle_allow", "brake_allow", "left_allow", "right_allow",
                              "throttle_forbid", "brake_forbid", "left_forbid", "right_forbid"]):
        if sample.get(name):
            out |= 1 << i
    out |= pack_field(int(sample.get("reason", 0)), 8, 6)
    if sample.get("action_active"):
        out |= 1 << 14
    return u32(out)


def hsst_decode(payload: int) -> dict[str, Any]:
    return {
        "left_count": extract(payload, 0, 8),
        "right_count": extract(payload, 8, 8),
        "left_type": extract(payload, 16, 2),
        "right_type": extract(payload, 18, 2),
        "left_count_valid": bit(payload, 20),
        "right_count_valid": bit(payload, 21),
        "left_type_valid": bit(payload, 22),
        "right_type_valid": bit(payload, 23),
        "reserved": extract(payload, 24, 8),
        "note": "counts are boundary counts, not lane counts",
    }


def nav_caps(caps: int) -> bool:
    return bool(caps & NAV_CAP)


GOLDEN = {
    "hud_left_adas": {"offset": OFF_HUD_REQUEST, "value": 0x00011103,
                      "name": "HUD left+ADAS seq1"},
    "hud_release": {"offset": OFF_HUD_REQUEST, "value": 0x00020000,
                    "name": "HUD release takeover seq2"},
    "scene": {"writes": [(OFF_SCENE_SHADOW, 0x00000985),
                         (OFF_SCENE_TIME0, 0x10242026),
                         (OFF_SCENE_TIME1, 0x00053014),
                         (OFF_SCENE_COMMIT, 0x0000002A)],
              "name": "scene seq42"},
    "nav_left_120": {"writes": [(OFF_NAV_SHADOW, 0x00780003),
                                (OFF_NAV_CONTROL, 0x00010001)],
                     "name": "nav left 120m seq1"},
}
