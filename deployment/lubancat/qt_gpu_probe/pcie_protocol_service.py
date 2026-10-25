#!/usr/bin/env python3
"""RK3566 PCIe protocol v1.0.1 control worker and browser debug console.

One process owns the BAR1 control window.  The browser console and the Qt
ISP/ALGO client both use this service; there is no second BAR1 writer.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent
SPEC = json.loads((BASE / "rk3566_pcie_protocol_v1.json").read_text(encoding="utf-8"))
MAGIC = 0x31435049
HUD_MAGIC = 0x31445548
DEFAULT_BAR1 = "/sys/bus/pci/devices/0000:01:00.0/resource1"

HUD = {"magic": 0x600, "version": 0x604, "request": 0x608, "status": 0x60C,
       "accepted": 0x610, "applied": 0x614, "rk_state": 0x618,
       "effective": 0x61C, "reject": 0x620, "last_error": 0x624,
       "rejected_seq": 0x628, "snapshot": 0x62C, "caps": 0x630,
       "last_request": 0x634}
SCENE = {"shadow": 0x638, "time0": 0x63C, "time1": 0x640, "commit": 0x644,
         "cdc": 0x648, "applied": 0x64C, "status": 0x658}
NAV = {"shadow": 0x670, "control": 0x674, "status": 0x678, "active": 0x67C}
HSST = {"sequence": 0x680, "payload": 0x684, "status": 0x688, "protocol": 0x68C}
BEHAVIOR = {"result": 0x65C, "effective": 0x660, "caps": 0x664}
ISP = {"global": 0x70C, "tile": 0x710, "commit": 0x750,
       "status": 0x754, "result": 0x758, "clear": 0x75C}
ALGO = {"lo": 0x760, "hi": 0x764, "commit": 0x768,
        "status": 0x76C, "result": 0x770, "clear": 0x774,
        "snap_index": 0x780, "snap_data": 0x784, "snap_lo": 0x788, "snap_hi": 0x78C}
VIDEO = {"magic": 0x400, "version": 0x404, "command": 0x408, "status": 0x40C,
         "generation": 0x410, "mode": 0x414, "rate": 0x418,
         "done": 0x41C, "error": 0x420, "drop": 0x424, "overflow": 0x428}
LEGACY = {"h2f_done": 0x130, "f2h_done": 0x230}
DESC_BASE = 0x500
DMA_BLOCKED = "VIDEO_DMA_CONTROL_UNAVAILABLE"


def u32(v):
    return int(v) & 0xFFFFFFFF


def u64(v):
    return int(v) & 0xFFFFFFFFFFFFFFFF


def hex32(v):
    return f"0x{u32(v):08x}"


def hex64(v):
    return f"0x{u64(v):016x}"


def decode_fields(value, fields):
    out = {}
    for f in fields:
        lsb, width = int(f["lsb"]), int(f["width_bits"])
        raw = (value >> lsb) & ((1 << width) - 1)
        if f.get("signed") and raw & (1 << (width - 1)):
            raw -= 1 << width
        out[f["name"]] = raw
    return out


def encode_fields(value, fields, values):
    for f in fields:
        name = f["name"]
        if name not in values:
            continue
        lsb, width = int(f["lsb"]), int(f["width_bits"])
        v = int(values[name])
        mask = (1 << width) - 1
        if f.get("signed"):
            v &= mask
        elif not (0 <= v <= mask):
            raise ValueError(f"{name} out of range 0..{mask}")
        value = (value & ~(mask << lsb)) | ((v & mask) << lsb)
    return value


def bcd_byte(v, name):
    v = int(v)
    if not (0 <= v <= 99):
        raise ValueError(f"{name} must be 0..99")
    return ((v // 10) << 4) | (v % 10)


def bcd_time0(dt):
    return (bcd_byte(dt.month, "month") << 24) | (bcd_byte(dt.day, "day") << 16) | \
           (bcd_byte(dt.year % 100, "year") << 8) | bcd_byte(dt.year // 100, "century")


def bcd_time1(dt):
    return (bcd_byte(dt.second, "second") << 16) | (bcd_byte(dt.minute, "minute") << 8) | \
           bcd_byte(dt.hour, "hour")


class Backend:
    def __init__(self, resource: str, simulation: bool = False, allow_write: bool = True):
        self.resource = resource
        self.lock = threading.RLock()
        self.logs = deque(maxlen=500)
        self.fd = None
        self.mm = None
        self.mode = "DISCONNECTED"
        self.available = False
        self.error = ""
        self.allow_write = allow_write
        self.sim = bool(simulation)
        self.hud_seq = 1
        self.scene_seq = 0
        self.nav_seq = 0
        self.isp_seq = 0
        self.algo_seq = 0
        self.hud = {"accepted": 0, "applied": 0, "effective": 0x101,
                    "reject": 0, "last_error": 0, "last_request": 0}
        self.scene = {"seq": 0, "applied": 0, "shadow": 0, "time0": 0,
                      "time1": 0, "status": 0}
        self.nav = {"seq": 0, "shadow": 0, "active": 0}
        self.isp_global = 0x1E
        reset_dwords = [int(x, 0) for x in SPEC["image_control"]["isp_configuration"]
                        ["tile_reset_dwords_low_to_high"]]
        self.isp_tiles = [list(reset_dwords) for _ in range(4)]
        self.algo = int(SPEC["image_control"]["algo_configuration"]["reset_hex64"], 16)
        self.isp_epoch = 0
        self.isp_source = 0
        self.hsst = {"sequence": 0x2A, "payload": 0x00F90102, "status": 0,
                     "protocol": 0x4C4D0001}
        self.video = {"magic": 0x31564750, "version": 1, "status": 0,
                      "mode": 0, "generation": 1, "rate": 0}
        if self.sim:
            self.mode = "SIMULATION"
            self.available = True
            self.error = "explicit simulation mode"
        else:
            self._open()

    def _open(self):
        try:
            self.fd = os.open(self.resource, os.O_RDWR | getattr(os, "O_SYNC", 0))
            import mmap
            self.mm = mmap.mmap(self.fd, 4096, access=mmap.ACCESS_WRITE)
            if self.rd(ISP["global"]) != MAGIC:
                self.mode = "DISCONNECTED"
                self.error = "BAR1 magic mismatch"
                return
            self.mode = "HARDWARE"
            self.available = True
            self.error = ""
        except Exception as exc:
            self.mode = "DISCONNECTED"
            self.available = False
            self.error = f"{type(exc).__name__}: {exc}"

    def rd(self, off):
        if self.sim:
            return self._sim_rd(off)
        if not self.mm:
            raise RuntimeError("BAR1 not mapped")
        return ctypes.c_uint32.from_buffer(self.mm, off).value

    def wr(self, off, value):
        if self.sim:
            self._sim_wr(off, value)
            return
        if not self.allow_write:
            raise RuntimeError("writes disabled")
        if not self.mm:
            raise RuntimeError("BAR1 not mapped")
        ctypes.c_uint32.from_buffer(self.mm, off).value = u32(value)

    def _log(self, op, offset, value, status="", note=""):
        self.logs.append({"t": datetime.now().isoformat(timespec="milliseconds"),
                          "op": op, "offset": hex32(offset), "value": hex32(value),
                          "status": status, "note": note})

    def _sim_rd(self, off):
        if off == 0x700:
            return MAGIC
        if off == 0x704:
            return 1
        if off == 0x708:
            return 0xF
        if off == HUD["magic"]:
            return HUD_MAGIC
        if off == HUD["version"]:
            return 0x00010000
        if off == HUD["caps"]:
            return 0x1FF
        if off == HUD["accepted"]:
            return self.hud["accepted"]
        if off == HUD["applied"]:
            return self.hud["applied"]
        if off == HUD["effective"]:
            return self.hud["effective"]
        if off == HUD["reject"]:
            return self.hud["reject"]
        if off == HUD["last_error"]:
            return self.hud["last_error"]
        if off == HUD["last_request"]:
            return self.hud["last_request"]
        if off == HUD["status"]:
            return 0xD8
        if off == HUD["rk_state"]:
            return self.hud["applied"] & 0x1FFF
        if off == HUD["rejected_seq"]:
            return 0
        if off == HUD["snapshot"]:
            return 1
        if off == SCENE["applied"]:
            return self.scene["applied"]
        if off == SCENE["commit"]:
            return self.scene["seq"]
        if off == SCENE["status"]:
            return (self.scene["seq"] << 16) | 0x0E
        if off == SCENE["cdc"]:
            return 2
        if off == SCENE["shadow"]:
            return self.scene["shadow"]
        if off == SCENE["time0"]:
            return self.scene["time0"]
        if off == SCENE["time1"]:
            return self.scene["time1"]
        if off == NAV["status"]:
            return 0x0B
        if off == NAV["control"]:
            return ((self.nav["seq"] & 0xFFFF) << 16) | (self.nav["seq"] & 0xFFFF)
        if off == NAV["active"]:
            return self.nav["active"]
        if off == BEHAVIOR["result"] or off == BEHAVIOR["effective"]:
            return 0x00014000 | (self.scene["seq"] & 0xFF) << 24
        if off == BEHAVIOR["caps"]:
            return 0x0002003F
        if off == HSST["sequence"]:
            return self.hsst["sequence"]
        if off == HSST["payload"]:
            return self.hsst["payload"]
        if off == HSST["status"]:
            return self.hsst["status"]
        if off == HSST["protocol"]:
            return self.hsst["protocol"]
        if off == ISP["global"]:
            return self.isp_global
        if ISP["tile"] <= off < ISP["tile"] + 64:
            idx = (off - ISP["tile"]) // 16
            word = ((off - ISP["tile"]) % 16) // 4
            return self.isp_tiles[idx][word]
        if off in (ISP["commit"],):
            return (self.isp_seq & 0xFFFF) << 16
        if off in (ISP["status"],):
            return 0x00020000 | ((self.isp_seq & 0xFFFF) << 16)
        if off == ISP["result"]:
            return ((self.isp_source & 3) << 16) | (self.isp_epoch & 0xFFFF)
        if off == ALGO["lo"]:
            return self.algo & 0xFFFFFFFF
        if off == ALGO["hi"]:
            return (self.algo >> 32) & 0xFFFFFFFF
        if off == ALGO["status"]:
            return 0x00020000 | ((self.algo_seq & 0xFFFF) << 16)
        if off == ALGO["result"]:
            return ((self.isp_source & 3) << 16) | (self.isp_epoch & 0xFFFF)
        if off in (VIDEO["magic"],):
            return 0x31564750
        if off in (VIDEO["version"],):
            return 1
        if off in (VIDEO["status"],):
            return 0
        if off in (VIDEO["mode"],):
            return 0
        if off in (VIDEO["generation"],):
            return 1
        if VIDEO["done"] <= off <= VIDEO["overflow"]:
            return 0
        return 0

    def _sim_wr(self, off, value):
        value = u32(value)
        if off == HUD["request"]:
            self._sim_hud(value)
        elif off in (SCENE["shadow"], SCENE["time0"], SCENE["time1"]):
            if off == SCENE["shadow"]:
                self.scene["shadow"] = value
            elif off == SCENE["time0"]:
                self.scene["time0"] = value
            else:
                self.scene["time1"] = value
        elif off == SCENE["commit"]:
            self.scene["seq"] = value & 0xFFFF
            self.scene["applied"] = self.scene["seq"]
        elif off == NAV["shadow"]:
            self.nav["shadow"] = value
        elif off == NAV["control"]:
            if value & 2:
                pass
            elif value & 1:
                self.nav["seq"] = (value >> 16) & 0xFFFF
                self.nav["active"] = self.nav["shadow"]
        elif off == ISP["global"]:
            self.isp_global = value & 0xFF
        elif ISP["tile"] <= off < ISP["tile"] + 64:
            idx = (off - ISP["tile"]) // 16
            word = ((off - ISP["tile"]) % 16) // 4
            self.isp_tiles[idx][word] = value
        elif off in (ISP["commit"], ALGO["commit"]):
            self.isp_seq = (value >> 16) & 0xFFFF
            self.isp_source = 2
            self.isp_epoch = (self.isp_epoch + 1) & 0xFFFF
            if off == ALGO["commit"]:
                self.algo_seq = self.isp_seq
        elif off == ALGO["lo"]:
            self.algo = (self.algo & 0xFFFFFFFF00000000) | value
        elif off == ALGO["hi"]:
            self.algo = (self.algo & 0xFFFFFFFF) | (value << 32)
        else:
            raise ValueError(f"write to non-whitelisted register 0x{off:03x}")

    def _sim_hud(self, value):
        seq = (value >> 16) & 0xFFFF
        payload = value & 0x1FFF
        if seq == 0:
            self.hud["reject"] += 1
            self.hud["last_error"] = 5
            return
        if payload & 0xE00:
            if (payload & 0x200 and not payload & 0x2) or \
               (payload & 0x400 and not payload & 0x4) or \
               (payload & 0x800 and not payload & 0x8):
                self.hud["reject"] += 1
                self.hud["last_error"] = 4
                return
        if seq == self.hud["accepted"] and value != self.hud["last_request"]:
            self.hud["reject"] += 1
            self.hud["last_error"] = 6
            return
        self.hud["accepted"] = seq
        self.hud["applied"] = seq
        self.hud["last_request"] = value
        self.hud["effective"] = (payload & 0x1FF) or 0x101

    def caps(self):
        if self.sim:
            return {"mode": self.mode, "available": True, "error": self.error,
                    "magic": hex32(MAGIC), "version": 1, "caps": "0x0000000f"}
        if not self.available:
            return {"mode": self.mode, "available": False, "error": self.error}
        try:
            return {"mode": self.mode, "available": True, "error": "",
                    "magic": hex32(self.rd(0x700)), "version": self.rd(0x704),
                    "caps": hex32(self.rd(0x708))}
        except Exception as exc:
            return {"mode": "DISCONNECTED", "available": False,
                    "error": f"{type(exc).__name__}: {exc}"}

    def _unavailable(self):
        return {"ok": False, "state": "DISCONNECTED", "mode": self.mode,
                "error": self.error or "BAR1 unavailable"}

    def _next(self, current, allow_zero=False):
        n = (current + 1) & 0xFFFF
        if not allow_zero and n == 0:
            n = 1
        return n

    def hud_apply(self, payload):
        with self.lock:
            if not self.available:
                return self._unavailable()
            values = dict(payload.get("fields", {}))
            seq = int(payload.get("sequence", 0)) & 0xFFFF
            if seq == 0:
                seq = self._next(self.hud["accepted"])
            value = 0
            for f in SPEC["hud_basic"]["request_fields"]:
                if f["name"] == "sequence":
                    value |= seq << int(f["lsb"])
                    continue
                if f["name"] in ("reserved",):
                    continue
                if f["name"] in values:
                    v = int(values[f["name"]]) & 1
                    value |= v << int(f["lsb"])
            for en, blink in (("left", "left_blink"), ("right", "right_blink"),
                              ("hazard", "hazard_blink")):
                if value & (1 << next(f["lsb"] for f in SPEC["hud_basic"]["request_fields"]
                                      if f["name"] == blink)) and not value & (1 << next(
                                          f["lsb"] for f in SPEC["hud_basic"]["request_fields"]
                                          if f["name"] == en)):
                    if self.sim:
                        self.hud["reject"] += 1
                        self.hud["last_error"] = 4
                    return {"ok": False, "state": "REJECTED", "error": f"{blink} requires {en}",
                            "last_error": self.hud["last_error"]}
            before = self.hud["last_error"]
            if self.sim:
                self._sim_hud(value)
                ok = self.hud["last_error"] == before
            else:
                old_reject = self.rd(HUD["reject"])
                self._log("HUD WRITE", HUD["request"], value, "WROTE")
                self.wr(HUD["request"], value)
                end = time.monotonic() + 1.0
                ok = False
                while time.monotonic() < end:
                    st = self.rd(HUD["status"])
                    err = (st >> 8) & 0xFF
                    if st & 4:
                        return {"ok": False, "state": "REJECTED", "status": hex32(st),
                                "error": err, "reject_count": self.rd(HUD["reject"]),
                                "last_error": self.rd(HUD["last_error"]),
                                "rejected_seq": self.rd(HUD["rejected_seq"])}
                    if not (st & 1) and self.rd(HUD["accepted"]) == seq and \
                       self.rd(HUD["applied"]) == seq:
                        ok = True
                        break
                    time.sleep(0.002)
                if not ok:
                    return {"ok": False, "state": "TIMEOUT_UNCONFIRMED",
                            "status": hex32(self.rd(HUD["status"])),
                            "accepted": self.rd(HUD["accepted"]),
                            "applied": self.rd(HUD["applied"])}
                self.hud["accepted"] = seq
                self.hud["applied"] = seq
                self.hud["effective"] = self.rd(HUD["effective"])
                self.hud["reject"] = self.rd(HUD["reject"])
                self.hud["last_error"] = self.rd(HUD["last_error"])
                self.hud["last_request"] = self.rd(HUD["last_request"])
                if self.hud["reject"] != old_reject and self.hud["last_error"] == 0:
                    ok = False
            result = {"ok": ok, "state": "APPLIED" if ok else "REJECTED",
                      "sequence": seq, "request": hex32(value),
                      "status": hex32(self.rd(HUD["status"])),
                      "accepted": self.rd(HUD["accepted"]),
                      "applied": self.rd(HUD["applied"]),
                      "effective": hex32(self.rd(HUD["effective"])),
                      "reject_count": self.rd(HUD["reject"]),
                      "last_error": self.rd(HUD["last_error"]),
                      "last_accepted_request": hex32(self.rd(HUD["last_request"]))}
            self.logs.append({"t": datetime.now().isoformat(timespec="milliseconds"),
                              "op": "HUD APPLY", "offset": "0x608", "value": hex32(value),
                              "status": result["state"]})
            return result

    def hud_read(self):
        with self.lock:
            if not self.available:
                return self._unavailable()
            return {"ok": True, "state": "READ",
                    "status": hex32(self.rd(HUD["status"])),
                    "accepted": self.rd(HUD["accepted"]),
                    "applied": self.rd(HUD["applied"]),
                    "rk_state": hex32(self.rd(HUD["rk_state"])),
                    "effective": hex32(self.rd(HUD["effective"])),
                    "reject_count": self.rd(HUD["reject"]),
                    "last_error": self.rd(HUD["last_error"]),
                    "rejected_seq": self.rd(HUD["rejected_seq"]),
                    "caps": hex32(self.rd(HUD["caps"])),
                    "last_accepted_request": hex32(self.rd(HUD["last_request"]))}

    def _scene_encode(self, payload):
        shadow = 0
        fields = payload.get("fields", payload)
        for f in SPEC["scene"]["shadow_fields"]:
            if f["name"] in fields:
                v = int(fields[f["name"]])
                shadow |= (v & ((1 << int(f["width_bits"])) - 1)) << int(f["lsb"])
        if payload.get("time"):
            dt = datetime.fromisoformat(payload["time"].replace("Z", ""))
        else:
            dt = datetime.now()
        time0 = int(payload.get("time0", bcd_time0(dt)))
        time1 = int(payload.get("time1", bcd_time1(dt)))
        seq = int(payload.get("sequence", 0)) & 0xFFFF
        if seq == 0:
            seq = self._next(self.scene["seq"])
        return shadow, time0, time1, seq

    def scene_apply(self, payload):
        with self.lock:
            if not self.available:
                return self._unavailable()
            shadow, time0, time1, seq = self._scene_encode(payload)
            if self.sim:
                self.scene.update(shadow=shadow, time0=time0, time1=time1,
                                  seq=seq, applied=seq, status=(seq << 16) | 0x0E)
            else:
                for off, val in ((SCENE["shadow"], shadow), (SCENE["time0"], time0),
                                 (SCENE["time1"], time1), (SCENE["commit"], seq)):
                    self._log("SCENE WRITE", off, val, "WROTE")
                    self.wr(off, val)
                time.sleep(0.02)
            result = {"ok": True, "state": "ACCEPTED", "sequence": seq,
                      "shadow": hex32(shadow), "time0": hex32(time0),
                      "time1": hex32(time1), "commit": hex32(seq),
                      "received": self.rd(SCENE["commit"]) & 0xFFFF,
                      "consumer": (self.rd(SCENE["status"]) >> 16) & 0xFFFF,
                      "applied": self.rd(SCENE["applied"]) & 0xFFFF,
                      "status": hex32(self.rd(SCENE["status"]))}
            self.logs.append({"t": datetime.now().isoformat(timespec="milliseconds"),
                              "op": "SCENE COMMIT", "offset": "0x644",
                              "value": hex32(seq), "status": result["state"]})
            return result

    def scene_read(self):
        with self.lock:
            if not self.available:
                return self._unavailable()
            return {"ok": True, "state": "READ",
                    "shadow": hex32(self.rd(SCENE["shadow"])),
                    "time0": hex32(self.rd(SCENE["time0"])),
                    "time1": hex32(self.rd(SCENE["time1"])),
                    "commit": hex32(self.rd(SCENE["commit"])),
                    "cdc": hex32(self.rd(SCENE["cdc"])),
                    "applied": self.rd(SCENE["applied"]) & 0xFFFF,
                    "status": hex32(self.rd(SCENE["status"]))}

    def nav_apply(self, payload):
        with self.lock:
            if not self.available:
                return self._unavailable()
            enable = 1 if payload.get("enable", True) else 0
            direction = int(payload.get("direction", 0)) & 3
            distance = int(payload.get("distance", 0)) & 0xFFFF
            if direction > 2:
                return {"ok": False, "state": "REJECTED", "error": "invalid direction"}
            shadow = (distance << 16) | (direction << 1) | enable
            seq = int(payload.get("sequence", self._next(self.nav["seq"], True))) & 0xFFFF
            control = (seq << 16) | 1
            if self.sim:
                self.nav.update(seq=seq, shadow=shadow, active=shadow)
            else:
                self._log("NAV SHADOW", NAV["shadow"], shadow, "WROTE")
                self.wr(NAV["shadow"], shadow)
                self.wr(NAV["control"], 2)
                self._log("NAV APPLY", NAV["control"], control, "WROTE")
                self.wr(NAV["control"], control)
                end = time.monotonic() + 1.0
                while time.monotonic() < end:
                    st = self.rd(NAV["status"])
                    ctl = self.rd(NAV["control"])
                    if not (st & 1) and (st & 0x0A) == 0x0A and (ctl >> 16) == seq:
                        break
                    time.sleep(0.002)
            result = {"ok": True, "state": "APPLIED", "sequence": seq,
                      "shadow": hex32(shadow), "control": hex32(control),
                      "status": hex32(self.rd(NAV["status"])),
                      "active": hex32(self.rd(NAV["active"]))}
            self.logs.append({"t": datetime.now().isoformat(timespec="milliseconds"),
                              "op": "NAV APPLY", "offset": "0x674",
                              "value": hex32(control), "status": result["state"]})
            return result

    def nav_clear(self):
        with self.lock:
            if not self.available:
                return self._unavailable()
            self.wr(NAV["control"], 2)
            return {"ok": True, "state": "CLEARED", "status": hex32(self.rd(NAV["status"]))}

    def nav_read(self):
        with self.lock:
            if not self.available:
                return self._unavailable()
            return {"ok": True, "state": "READ",
                    "shadow": hex32(self.rd(NAV["shadow"])),
                    "control": hex32(self.rd(NAV["control"])),
                    "status": hex32(self.rd(NAV["status"])),
                    "active": hex32(self.rd(NAV["active"]))}

    def _isp_fields(self):
        return SPEC["image_control"]["isp_configuration"]["tile_fields"]

    def _tile_value(self, tile):
        a, b, c, d = tile
        return (d << 96) | (c << 64) | (b << 32) | a

    def _tile_dwords(self, value):
        return [value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF,
                (value >> 64) & 0xFFFFFFFF, (value >> 96) & 0xFFFFFFFF]

    def _read_tiles(self):
        if self.sim:
            return [list(t) for t in self.isp_tiles]
        return [[self.rd(ISP["tile"] + i * 16 + j * 4) for j in range(4)]
                for i in range(4)]

    def _transaction(self, channel, opcode):
        status_off = ISP["status"] if channel == "isp" else ALGO["status"]
        commit_off = ISP["commit"] if channel == "isp" else ALGO["commit"]
        clear_off = ISP["clear"] if channel == "isp" else ALGO["clear"]
        result_off = ISP["result"] if channel == "isp" else ALGO["result"]
        current = self.isp_seq if channel == "isp" else self.algo_seq
        seq = self._next(current, True)
        if self.sim:
            self.isp_seq = seq
            if channel == "algo":
                self.algo_seq = seq
            self.isp_source = 2
            self.isp_epoch = (self.isp_epoch + 1) & 0xFFFF
            return {"ok": True, "state": "APPLIED", "sequence": seq,
                    "status": hex32((seq << 16) | 2)}
        self.wr(clear_off, 1)
        self.wr(commit_off, (seq << 16) | opcode)
        self._log(channel.upper() + " COMMIT", commit_off, (seq << 16) | opcode, "WROTE")
        end = time.monotonic() + 3.0
        while time.monotonic() < end:
            st = self.rd(status_off)
            error = (st >> 8) & 0xFF
            if st & 4:
                return {"ok": False, "state": "REJECTED", "sequence": seq,
                        "status": hex32(st), "error": error}
            if not (st & 1) and (st & 2) and (st >> 16) == seq:
                res = self.rd(result_off)
                self.isp_seq = seq
                if channel == "algo":
                    self.algo_seq = seq
                return {"ok": True, "state": "APPLIED", "sequence": seq,
                        "status": hex32(st), "epoch": res & 0xFFFF,
                        "source": (res >> 16) & 3}
            time.sleep(0.002)
        return {"ok": False, "state": "TIMEOUT_UNCONFIRMED", "sequence": seq,
                "status": hex32(self.rd(status_off))}

    def isp_query(self):
        with self.lock:
            if not self.available:
                return self._unavailable()
            if not self.sim:
                tx = self._transaction("isp", 2)
                if not tx.get("ok") and tx.get("state") != "APPLIED":
                    return tx
            values = self._read_tiles()
            return {"ok": True, "state": "READ", "global": hex32(self.isp_global),
                    "tiles": [[hex32(v) for v in tile] for tile in values],
                    "tile_fields": [decode_fields(self._tile_value(tile), self._isp_fields())
                                    for tile in values],
                    "algo": hex64(self.algo),
                    "epoch": self.isp_epoch, "source": self.isp_source}

    def isp_apply(self, payload):
        with self.lock:
            if not self.available:
                return self._unavailable()
            global_value = payload.get("global")
            if global_value is not None:
                global_value = int(str(global_value), 0) & 0xFF
            else:
                global_value = self.isp_global
            incoming = payload.get("tiles")
            if not isinstance(incoming, list) or len(incoming) != 4:
                return {"ok": False, "state": "REJECTED", "error": "four tiles required"}
            values = self._read_tiles()
            for i, tile in enumerate(incoming):
                if isinstance(tile, dict):
                    raw = self._tile_value(values[i])
                    raw = encode_fields(raw, self._isp_fields(), tile)
                    if raw >> 128:
                        return {"ok": False, "state": "REJECTED", "error": "tile overflow"}
                    values[i] = self._tile_dwords(raw)
                elif isinstance(tile, list) and len(tile) == 4:
                    values[i] = [int(str(x), 0) & 0xFFFFFFFF for x in tile]
                else:
                    return {"ok": False, "state": "REJECTED", "error": "tile needs 4 dwords or fields"}
            if self.sim:
                self.isp_global = global_value
                self.isp_tiles = values
            else:
                self.wr(ISP["global"], global_value)
                for i, tile in enumerate(values):
                    for j, value in enumerate(tile):
                        self.wr(ISP["tile"] + i * 16 + j * 4, value)
            tx = self._transaction("isp", 1)
            if tx.get("ok"):
                state = tx.get("state")
                tx.update(self.isp_query())
                tx["state"] = state
            self.logs.append({"t": datetime.now().isoformat(timespec="milliseconds"),
                              "op": "ISP APPLY", "offset": "0x70c-0x74c",
                              "value": "tiles", "status": tx.get("state", "")})
            return tx

    def algo_query(self):
        with self.lock:
            if not self.available:
                return self._unavailable()
            if not self.sim:
                self._transaction("algo", 2)
            fields = SPEC["image_control"]["algo_configuration"]["fields"]
            return {"ok": True, "state": "READ", "value": hex64(self.algo),
                    "fields": decode_fields(self.algo, fields),
                    "epoch": self.isp_epoch, "source": self.isp_source}

    def algo_apply(self, payload):
        with self.lock:
            if not self.available:
                return self._unavailable()
            fields = SPEC["image_control"]["algo_configuration"]["fields"]
            value = int(str(payload.get("value", hex64(self.algo))), 0)
            if isinstance(payload.get("fields"), dict):
                value = encode_fields(value, fields, payload["fields"])
            value = u64(value)
            if self.sim:
                self.algo = value
            else:
                self.wr(ALGO["lo"], value & 0xFFFFFFFF)
                self.wr(ALGO["hi"], value >> 32)
            tx = self._transaction("algo", 1)
            if tx.get("ok"):
                state = tx.get("state")
                tx.update(self.algo_query())
                tx["state"] = state
            self.logs.append({"t": datetime.now().isoformat(timespec="milliseconds"),
                              "op": "ALGO APPLY", "offset": "0x760/0x764",
                              "value": hex64(value), "status": tx.get("state", "")})
            return tx

    def diag(self):
        with self.lock:
            if not self.available:
                return self._unavailable()
            hsst = {k: self.rd(HSST[k]) for k in HSST}
            payload = hsst["payload"]
            result = self.rd(BEHAVIOR["result"])
            return {"ok": True, "state": "READ",
                    "hsst": {"sequence": hsst["sequence"], "payload": hex32(payload),
                             "protocol": hex32(hsst["protocol"]),
                             "status": hex32(hsst["status"]),
                             "valid": bool(hsst["status"] & 1),
                             "stale": bool(hsst["status"] & 2),
                             "left_count": payload & 0xFF,
                             "right_count": (payload >> 8) & 0xFF,
                             "left_type": (payload >> 16) & 3,
                             "right_type": (payload >> 18) & 3,
                             "left_count_valid": bool(payload & (1 << 20)),
                             "right_count_valid": bool(payload & (1 << 21)),
                             "left_type_valid": bool(payload & (1 << 22)),
                             "right_type_valid": bool(payload & (1 << 23))},
                    "behavior": {"result": hex32(result),
                                 "action_active": bool(result & (1 << 14)),
                                 "stale": bool(result & (1 << 15)),
                                 "reason": (result >> 8) & 0x3F,
                                 "throttle_allow": bool(result & 1),
                                 "brake_allow": bool(result & 2),
                                 "left_allow": bool(result & 4),
                                 "right_allow": bool(result & 8),
                                 "throttle_forbid": bool(result & 16),
                                 "brake_forbid": bool(result & 32),
                                 "left_forbid": bool(result & 64),
                                 "right_forbid": bool(result & 128),
                                 "action_seq": result & 0xFF,
                                 "scene_seq": (result >> 24) & 0xFF},
                    "video": {k: hex32(self.rd(VIDEO[k])) for k in VIDEO},
                    "legacy": {k: hex32(self.rd(v)) for k, v in LEGACY.items()},
                    "dma_write": "BLOCKED",
                    "dma_note": DMA_BLOCKED}

    def raw_hud(self, value):
        with self.lock:
            if not self.available:
                return self._unavailable()
            seq = (int(value) >> 16) & 0xFFFF
            return self.hud_apply({"fields": {}, "sequence": seq, "raw": value}) \
                if False else self._raw_write_hud(value)

    def _raw_write_hud(self, value):
        value = int(value) & 0xFFFFFFFF
        if self.sim:
            self._sim_hud(value)
        else:
            self._log("HUD VECTOR", HUD["request"], value, "WROTE")
            self.wr(HUD["request"], value)
        return {**self.hud_read(), "ok": True, "state": "SENT",
                "request": hex32(value)}

    def raw_scene(self, payload):
        with self.lock:
            if not self.available:
                return self._unavailable()
            writes = payload.get("writes", {})
            allowed = {"0x638": SCENE["shadow"], "0x63c": SCENE["time0"],
                       "0x640": SCENE["time1"], "0x644": SCENE["commit"]}
            for key, value in writes.items():
                if key.lower() not in allowed:
                    return {"ok": False, "state": "REJECTED", "error": f"bad scene offset {key}"}
                self.wr(allowed[key.lower()], int(str(value), 0))
            return {**self.scene_read(), "ok": True, "state": "SENT"}


class Timer:
    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None

    def start(self, interval, fn):
        self.stop()
        self.stop_event.clear()
        def run():
            while not self.stop_event.wait(interval):
                try:
                    fn()
                except Exception as exc:
                    print("scene timer error", exc, flush=True)
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None


class Handler(BaseHTTPRequestHandler):
    backend = None
    token = ""

    def _auth(self):
        if not self.token:
            return True
        supplied = self.headers.get("X-Token", "") or \
            self.headers.get("X-Auth-Token", "")
        if not supplied:
            from urllib.parse import urlparse, parse_qs
            supplied = parse_qs(urlparse(self.path).query).get("token", [""])[0]
        if supplied == self.token:
            return True
        self.send_error(401, "token required")
        return False

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, name):
        path = BASE / name
        if not path.exists():
            self.send_error(404)
            return
        data = path.read_bytes()
        ctype = "text/html; charset=utf-8" if name.endswith(".html") else \
                "text/css" if name.endswith(".css") else "application/javascript"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._auth():
            return
        from urllib.parse import urlparse
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._file("pcie_debug_ui.html")
        if path == "/api/schema":
            return self._json(SPEC)
        if path == "/api/state":
            return self._json(self.backend.state())
        if path == "/api/log":
            return self._json({"logs": list(self.backend.logs)})
        if path == "/api/report":
            return self._json(self.backend.report())
        self.send_error(404)

    def do_POST(self):
        if not self._auth():
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            op = req.get("op")
            b = self.backend
            if op == "hud_apply":
                ans = b.hud_apply(req)
            elif op == "hud_read":
                ans = b.hud_read()
            elif op == "hud_raw":
                ans = b.raw_hud(req.get("value", 0))
            elif op == "scene_apply":
                ans = b.scene_apply(req)
            elif op == "scene_read":
                ans = b.scene_read()
            elif op == "scene_raw":
                ans = b.raw_scene(req)
            elif op == "nav_apply":
                ans = b.nav_apply(req)
            elif op == "nav_clear":
                ans = b.nav_clear()
            elif op == "nav_read":
                ans = b.nav_read()
            elif op == "isp_query":
                ans = b.isp_query()
            elif op == "isp_apply":
                ans = b.isp_apply(req)
            elif op == "algo_query":
                ans = b.algo_query()
            elif op == "algo_apply":
                ans = b.algo_apply(req)
            elif op == "diag":
                ans = b.diag()
            elif op == "caps":
                ans = {"ok": True, **b.caps()}
            else:
                ans = {"ok": False, "state": "REJECTED", "error": "unknown op"}
            self._json(ans)
        except Exception as exc:
            self._json({"ok": False, "state": "REJECTED",
                        "error": f"{type(exc).__name__}: {exc}"}, 400)


class DebugBackend(Backend):
    def state(self):
        return {"caps": self.caps(), "hud": self.hud_read(),
                "scene": self.scene_read(), "nav": self.nav_read(),
                "isp": self.isp_query(), "algo": self.algo_query(),
                "diag": self.diag(), "logs": list(self.logs),
                "server": {"mode": self.mode, "available": self.available,
                           "error": self.error}}

    def report(self):
        return {"generated": datetime.now().isoformat(), "caps": self.caps(),
                "state": self.state(), "logs": list(self.logs)}


def udp_worker(backend, port, stop):
    import select
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(0.5)
    print(f"UDP_READY port={port} mode={backend.mode}", flush=True)
    while not stop.is_set():
        try:
            ready, _, _ = select.select([sock], [], [], 0.5)
            if not ready:
                continue
            data, peer = sock.recvfrom(65535)
            req = json.loads(data)
            typ = req.get("type")
            if typ == "caps":
                reply = {"type": "caps", "ok": True, **backend.caps()}
            elif typ == "query":
                reply = {"type": "query", **backend.isp_query()}
            elif typ == "apply_isp":
                reply = {"type": "apply_isp", **backend.isp_apply(req)}
            elif typ == "apply_algo":
                reply = {"type": "apply_algo", **backend.algo_apply(req)}
            elif typ == "algo_query":
                reply = {"type": "algo_query", **backend.algo_query()}
            else:
                reply = {"type": "error", "ok": False, "error": "unknown request"}
            reply.setdefault("mode", backend.mode)
            reply.setdefault("available", backend.available)
            sock.sendto(json.dumps(reply).encode(), peer)
        except Exception as exc:
            print("udp error", exc, flush=True)
    sock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar1", default=DEFAULT_BAR1)
    ap.add_argument("--port", type=int, default=38264)
    ap.add_argument("--web-port", type=int, default=8080)
    ap.add_argument("--web-host", default="127.0.0.1")
    ap.add_argument("--token", default="")
    ap.add_argument("--simulation", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    backend = DebugBackend(args.bar1, simulation=args.simulation,
                           allow_write=not args.no_write)
    Handler.backend = backend
    Handler.token = args.token
    stop = threading.Event()
    udp = threading.Thread(target=udp_worker, args=(backend, args.port, stop), daemon=True)
    udp.start()
    server = ThreadingHTTPServer((args.web_host, args.web_port), Handler)
    print(f"WEB_READY http://{args.web_host}:{args.web_port}/ mode={backend.mode}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
