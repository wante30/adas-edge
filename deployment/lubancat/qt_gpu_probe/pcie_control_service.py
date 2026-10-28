#!/usr/bin/env python3
"""Unified RK3566 PCIe v1.0.1 control worker + local web console.

One process owns BAR1.  It never probes/writes video DMA and never falls back
from a failed hardware probe to simulation.  Simulation is explicit only.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import select
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pcie_protocol as P

HERE = Path(__file__).resolve().parent


def u32(v: Any) -> int:
    return int(v) & 0xFFFFFFFF


def hx(v: int, width: int = 8) -> str:
    return f"0x{int(v) & ((1 << width) - 1):0{width // 4}x}"


@dataclass
class SceneState:
    seq: int = 0
    accepted: int = 0
    consumed: int = 0
    applied: int = 0
    shadow: int = 0
    time0: int = 0
    time1: int = 0
    applied_ns: int = 0
    status: int = 0
    error: int = 0


@dataclass
class NavState:
    seq: int = 0
    accepted: int = 0
    applied: int = 0
    shadow: int = 0
    active: int = 0
    status: int = 0
    error: int = 0


@dataclass
class ImageState:
    global_word: int = P.ISP_GLOBAL_RESET
    tiles: list[list[int]] = field(default_factory=lambda: [list(P.ISP_RESET_DW) for _ in range(4)])
    algo: int = P.ALGO_RESET
    epoch: int = 0
    source: int = 0
    last_error: int = 0
    reject: int = 0


class Service:
    def __init__(self, resource: str, simulation: bool = False):
        self.resource = resource
        self.mode = "DISCONNECTED"
        self.error = ""
        self.sim = False
        self.available = False
        self.fd = None
        self.mm = None
        self.lock = threading.RLock()
        self.logs: list[dict[str, Any]] = []
        self.hud_seq = 0
        self.hud_effective = 0x101
        self.hud_last_error = 0
        self.hud_reject = 0
        self.scene = SceneState()
        self.nav = NavState()
        self.image = ImageState()
        self.hsst_seq = 0
        self.hsst_payload = 0x00F90102
        self.hsst_status = 0x00000001
        if simulation:
            self.sim = True
            self.available = True
            self.mode = "SIMULATION"
            self.error = "explicit simulation mode; no BAR1 access"
        else:
            self._open(resource)

    def _open(self, resource: str) -> None:
        try:
            self.fd = os.open(resource, os.O_RDWR | getattr(os, "O_SYNC", 0))
            self.mm = __import__("mmap").mmap(self.fd, 4096, access=__import__("mmap").ACCESS_WRITE)
            hud = self._rd(P.OFF_HUD_MAGIC)
            isp = self._rd(P.OFF_ISP_MAGIC)
            if hud != P.MAGIC_HUD and isp != P.MAGIC_ISP:
                self.available = False
                self.mode = "DISCONNECTED"
                self.error = f"BAR1 identity mismatch hud={hx(hud)} isp={hx(isp)}"
                return
            self.available = True
            self.mode = "HARDWARE"
            self.error = ""
            self.hud_seq = self._rd(P.OFF_HUD_APPLIED) & 0xFFFF
            self.hud_effective = self._rd(P.OFF_HUD_EFFECTIVE)
            self.scene.applied = self._rd(P.OFF_SCENE_APPLIED) & 0xFFFF
            self.scene.seq = self.scene.applied
            self.nav.accepted = (self._rd(P.OFF_NAV_CONTROL) >> 16) & 0xFFFF
            self.nav.seq = self.nav.accepted
        except Exception as exc:
            self.available = False
            self.mode = "DISCONNECTED"
            self.error = f"{type(exc).__name__}: {exc}"

    def _rd(self, off: int) -> int:
        if self.mm is None:
            raise RuntimeError("BAR1 not mapped")
        return ctypes.c_uint32.from_buffer(self.mm, off).value

    def _wr(self, off: int, value: int) -> None:
        if self.mm is None:
            raise RuntimeError("BAR1 not mapped")
        ctypes.c_uint32.from_buffer(self.mm, off).value = u32(value)

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": self.available,
            "mode": self.mode,
            "available": self.available,
            "error": self.error,
            "protocol_version": P.VERSION,
            "resource": self.resource,
        }
        if self.available and not self.sim:
            with self.lock:
                try:
                    out.update({
                        "hud_magic": hx(self._rd(P.OFF_HUD_MAGIC)),
                        "hud_version": hx(self._rd(P.OFF_HUD_VERSION)),
                        "hud_caps": hx(self._rd(P.OFF_HUD_CAPS)),
                        "isp_magic": hx(self._rd(P.OFF_ISP_MAGIC)),
                        "isp_caps": hx(self._rd(P.OFF_ISP_CAPS)),
                        "video_status": hx(self._rd(0x40C)),
                        "hdma_status": hx(self._rd(0x120)),
                        "fdma_status": hx(self._rd(0x220)),
                    })
                except Exception as exc:
                    out["error"] = f"read failed: {exc}"
        elif self.sim:
            out.update({"hud_magic": hx(P.MAGIC_HUD), "hud_caps": hx(P.HUD_CAPS),
                        "isp_magic": hx(P.MAGIC_ISP), "isp_caps": "0x0000000f"})
        return out

    def _record(self, group: str, action: str, writes: list[tuple[int, int]],
                result: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "time": time.strftime("%H:%M:%S"), "group": group, "action": action,
            "writes": [{"offset": hx(off), "value": hx(val)} for off, val in writes],
            "result": result,
        }
        self.logs.append(entry)
        del self.logs[:-200]
        return entry

    def _unavailable(self) -> dict[str, Any]:
        return {"ok": False, "state": "DISCONNECTED", "mode": self.mode, "error": self.error}

    # ---------------------------------------------------------------- HUD
    def hud_apply(self, fields: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not self.available:
                return self._unavailable()
            seq = int(fields.get("sequence") or (self.hud_seq + 1)) & 0xFFFF
            if seq == 0:
                seq = 1
            takeover = not bool(fields.get("release_takeover", False))
            try:
                word = P.hud_request(seq, fields, takeover=takeover)
            except ValueError as exc:
                return {"ok": False, "state": "REJECTED", "error": str(exc), "sequence": seq}
            writes = [(P.OFF_HUD_REQUEST, word)]
            if self.sim:
                self.hud_seq = seq
                self.hud_effective = word & 0x1FF
                self.hud_last_error = 0
                status = 0x00000008 | 0x02 | 0x04 | 0x10 | 0x20 | 0x40
                res = {"ok": True, "state": "APPLIED", "sequence": seq, "request": hx(word),
                       "status": hx(status), "accepted_seq": seq, "applied_seq": seq,
                       "effective": hx(self.hud_effective), "effective_fields": P.hud_decode_effective(self.hud_effective),
                       "reject_count": self.hud_reject, "last_error": hx(self.hud_last_error)}
                self._record("HUD", "APPLY", writes, res)
                return res
            try:
                before_reject = self._rd(P.OFF_HUD_REJECT)
                self._wr(P.OFF_HUD_REQUEST, word)
                deadline = time.monotonic() + 0.5
                accepted = applied = status = 0
                while time.monotonic() < deadline:
                    status = self._rd(P.OFF_HUD_STATUS)
                    accepted = self._rd(P.OFF_HUD_ACCEPTED)
                    applied = self._rd(P.OFF_HUD_APPLIED)
                    if applied == seq and (status & 0x40):
                        break
                    time.sleep(0.002)
                effective = self._rd(P.OFF_HUD_EFFECTIVE)
                last_error = self._rd(P.OFF_HUD_ERROR)
                ok = applied == seq and accepted == seq and not bool(status & 0x04)
                res = {"ok": ok, "state": "APPLIED" if ok else "TIMEOUT_UNCONFIRMED",
                       "sequence": seq, "request": hx(word), "status": hx(status),
                       "accepted_seq": accepted, "applied_seq": applied,
                       "effective": hx(effective), "effective_fields": P.hud_decode_effective(effective),
                       "reject_count": self._rd(P.OFF_HUD_REJECT), "last_error": hx(last_error),
                       "historical_error_note": "LAST_ERROR is sticky; compare reject_count"}
                if ok:
                    self.hud_seq, self.hud_effective = seq, effective
                self._record("HUD", "APPLY", writes, res)
                return res
            except Exception as exc:
                return {"ok": False, "state": "DISCONNECTED", "error": str(exc)}

    def hud_read(self) -> dict[str, Any]:
        with self.lock:
            if not self.available:
                return self._unavailable()
            if self.sim:
                return {"ok": True, "state": "READ", "request": hx((self.hud_seq << 16) | 0x1103),
                        "status": "0x00000078", "accepted_seq": self.hud_seq,
                        "applied_seq": self.hud_seq, "effective": hx(self.hud_effective),
                        "reject_count": self.hud_reject, "last_error": hx(self.hud_last_error)}
            regs = {name: hx(self._rd(off)) for name, off in {
                "request": P.OFF_HUD_REQUEST, "status": P.OFF_HUD_STATUS,
                "accepted_seq": P.OFF_HUD_ACCEPTED, "applied_seq": P.OFF_HUD_APPLIED,
                "effective": P.OFF_HUD_EFFECTIVE, "reject_count": P.OFF_HUD_REJECT,
                "last_error": P.OFF_HUD_ERROR, "last_request": P.OFF_HUD_LAST_REQUEST}.items()}
            return {"ok": True, "state": "READ", **regs}

    # ---------------------------------------------------------------- SCENE
    def scene_apply(self, fields: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not self.available:
                return self._unavailable()
            try:
                shadow = P.scene_shadow(fields)
                time0 = int(str(fields.get("time0", "0x10242026")), 0)
                time1 = int(str(fields.get("time1", "0x00053014")), 0)
                if fields.get("valid", True) and fields.get("show_time", True):
                    good, err = P.bcd_validate(time0, time1)
                    if not good:
                        return {"ok": False, "state": "REJECTED", "error": err}
                seq = int(fields.get("sequence") or (self.scene.seq + 1)) & 0xFFFF
                if seq == 0:
                    seq = 1
                commit = seq & 0xFFFF
            except Exception as exc:
                return {"ok": False, "state": "REJECTED", "error": str(exc)}
            writes = [(P.OFF_SCENE_SHADOW, shadow), (P.OFF_SCENE_TIME0, time0),
                      (P.OFF_SCENE_TIME1, time1), (P.OFF_SCENE_COMMIT, commit)]
            if self.sim:
                now = time.monotonic_ns()
                self.scene.seq = seq
                self.scene.accepted = seq
                self.scene.shadow = shadow
                self.scene.time0, self.scene.time1 = time0, time1
                self.scene.consumed = seq
                if fields.get("valid", True):
                    self.scene.applied = seq
                    self.scene.applied_ns = now
                else:
                    self.scene.applied = 0
                    self.scene.applied_ns = 0
                res = {"ok": True, "state": "APPLIED" if fields.get("valid", True) else "CLEARED",
                       "sequence": seq, "shadow": hx(shadow), "commit": hx(commit),
                       "accepted_seq": seq, "consumed_seq": seq,
                       "applied_seq": self.scene.applied, "status": hx(self.scene.status),
                       "hold_ms": 500, "time0": hx(time0), "time1": hx(time1)}
                self._record("SCENE", "APPLY", writes, res)
                return res
            try:
                self._wr(P.OFF_SCENE_SHADOW, shadow)
                self._wr(P.OFF_SCENE_TIME0, time0)
                self._wr(P.OFF_SCENE_TIME1, time1)
                self._wr(P.OFF_SCENE_COMMIT, commit)
                deadline = time.monotonic() + 0.5
                status = applied = consumed = 0
                while time.monotonic() < deadline:
                    status = self._rd(P.OFF_SCENE_STATUS)
                    applied = self._rd(P.OFF_SCENE_APPLIED) & 0xFFFF
                    consumed = (status >> 16) & 0xFFFF
                    if applied == seq and consumed == seq:
                        break
                    time.sleep(0.002)
                res = {"ok": applied == seq and not bool(status & 0x10),
                       "state": "APPLIED" if applied == seq else "TIMEOUT_UNCONFIRMED",
                       "sequence": seq, "shadow": hx(shadow), "commit": hx(commit),
                       "accepted_seq": consumed, "consumed_seq": consumed,
                       "applied_seq": applied, "status": hx(status),
                       "hold_ms": 501, "time0": hx(time0), "time1": hx(time1)}
                self._record("SCENE", "APPLY", writes, res)
                return res
            except Exception as exc:
                return {"ok": False, "state": "DISCONNECTED", "error": str(exc)}

    def scene_read(self) -> dict[str, Any]:
        with self.lock:
            if not self.available:
                return self._unavailable()
            if self.sim:
                return {"ok": True, "state": "READ", "shadow": hx(self.scene.shadow),
                        "status": hx(self.scene.status), "applied_seq": self.scene.applied,
                        "consumed_seq": self.scene.consumed, "time0": hx(self.scene.time0),
                        "time1": hx(self.scene.time1)}
            regs = {name: hx(self._rd(off)) for name, off in {
                "shadow": P.OFF_SCENE_SHADOW, "time0": P.OFF_SCENE_TIME0, "time1": P.OFF_SCENE_TIME1,
                "status": P.OFF_SCENE_STATUS, "applied_seq": P.OFF_SCENE_APPLIED}.items()}
            return {"ok": True, "state": "READ", **regs}

    # ---------------------------------------------------------------- NAV
    def nav_apply(self, fields: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not self.available:
                return self._unavailable()
            try:
                shadow = P.nav_shadow(bool(fields.get("enabled", True)),
                                      str(fields.get("direction", "left")),
                                      int(fields.get("distance_m", 120)))
                seq = int(fields.get("sequence", (self.nav.seq + 1) & 0xFFFF)) & 0xFFFF
                control = P.nav_control(seq, True)
            except Exception as exc:
                return {"ok": False, "state": "REJECTED", "error": str(exc)}
            writes = [(P.OFF_NAV_SHADOW, shadow), (P.OFF_NAV_CONTROL, control)]
            if self.sim:
                self.nav.seq, self.nav.accepted, self.nav.applied = seq, seq, seq
                self.nav.shadow, self.nav.active, self.nav.status = shadow, shadow, 0x0E
                res = {"ok": True, "state": "APPLIED", "sequence": seq, "shadow": hx(shadow),
                       "control": hx(control), "accepted_seq": seq, "applied_seq": seq,
                       "active": hx(shadow), "status": hx(self.nav.status)}
                self._record("NAV", "APPLY", writes, res)
                return res
            try:
                status = self._rd(P.OFF_NAV_STATUS)
                if not (status & 0x08):
                    return {"ok": False, "state": "REJECTED", "error": "NAV not ready", "status": hx(status)}
                self._wr(P.OFF_NAV_CONTROL, 2)
                self._wr(P.OFF_NAV_SHADOW, shadow)
                self._wr(P.OFF_NAV_CONTROL, control)
                deadline = time.monotonic() + 0.5
                while time.monotonic() < deadline:
                    status = self._rd(P.OFF_NAV_STATUS)
                    ctl = self._rd(P.OFF_NAV_CONTROL)
                    active = self._rd(P.OFF_NAV_ACTIVE)
                    if (not (status & 1) and (status & 2) and (ctl >> 16) == seq
                            and (ctl & 0xFFFF) == seq and active == shadow and not (status & 0x20)):
                        res = {"ok": True, "state": "APPLIED", "sequence": seq,
                               "shadow": hx(shadow), "control": hx(control),
                               "accepted_seq": seq, "applied_seq": seq,
                               "active": hx(active), "status": hx(status)}
                        self._record("NAV", "APPLY", writes, res)
                        return res
                    time.sleep(0.002)
                res = {"ok": False, "state": "TIMEOUT_UNCONFIRMED", "sequence": seq,
                       "shadow": hx(shadow), "status": hx(status)}
                self._record("NAV", "APPLY", writes, res)
                return res
            except Exception as exc:
                return {"ok": False, "state": "DISCONNECTED", "error": str(exc)}

    def nav_read(self) -> dict[str, Any]:
        with self.lock:
            if not self.available:
                return self._unavailable()
            if self.sim:
                return {"ok": True, "state": "READ", "shadow": hx(self.nav.shadow),
                        "control": hx(P.nav_control(self.nav.seq, True)),
                        "status": hx(self.nav.status), "active": hx(self.nav.active),
                        "decoded": P.nav_decode(self.nav.active)}
            regs = {name: hx(self._rd(off)) for name, off in {
                "shadow": P.OFF_NAV_SHADOW, "control": P.OFF_NAV_CONTROL,
                "status": P.OFF_NAV_STATUS, "active": P.OFF_NAV_ACTIVE}.items()}
            return {"ok": True, "state": "READ", **regs}

    # ---------------------------------------------------------------- IMAGE
    def _image_status(self, seq: int, error: int = 0, busy: bool = False) -> int:
        return ((seq & 0xFFFF) << 16) | (error << 8) | (4 if error else 0) | (2 if not busy and not error else 0) | (1 if busy else 0)

    def isp_query(self) -> dict[str, Any]:
        with self.lock:
            if not self.available:
                return self._unavailable()
            if self.sim:
                self.image.epoch = (self.image.epoch + 1) & 0xFFFF
                self.image.source = 2
                res = {"ok": True, "state": "READ", "global": hx(self.image.global_word),
                       "tiles": [[hx(x) for x in t] for t in self.image.tiles],
                       "tile_fields": [P.isp_tile_decode(P.isp_words_to_value(t)) for t in self.image.tiles],
                       "algo": hx(self.image.algo, 16), "algo_fields": P.algo_decode(self.image.algo),
                       "algo_stages": P.algo_stages(self.image.algo),
                       "epoch": self.image.epoch, "source": self.image.source, "status": self._image_status(self.image.epoch)}
                self._record("ISP", "QUERY", [], res)
                return res
            try:
                self._wr(P.OFF_ISP_CLEAR, 1)
                seq = ((self._rd(P.OFF_ISP_COMMIT) >> 16) + 1) & 0xFFFF
                self._wr(P.OFF_ISP_COMMIT, (seq << 16) | 2)
                deadline = time.monotonic() + 1.0
                status = 0
                while time.monotonic() < deadline:
                    status = self._rd(P.OFF_ISP_STATUS)
                    if status & 0x04:
                        return {"ok": False, "state": "REJECTED", "status": hx(status), "error": (status >> 8) & 0xFF}
                    if not (status & 1) and (status & 2) and ((status >> 16) & 0xFFFF) == seq:
                        break
                    time.sleep(0.002)
                tiles = [[self._rd(P.OFF_ISP_TILE + i * 16 + j * 4) for j in range(4)] for i in range(4)]
                res = {"ok": True, "state": "READ", "global": hx(self._rd(P.OFF_ISP_GLOBAL)),
                       "tiles": [[hx(x) for x in t] for t in tiles],
                       "tile_fields": [P.isp_tile_decode(P.isp_words_to_value(t)) for t in tiles],
                       "algo": hx((self._rd(P.OFF_ALGO_HI) << 32) | self._rd(P.OFF_ALGO_LO), 16),
                       "algo_fields": P.algo_decode((self._rd(P.OFF_ALGO_HI) << 32) | self._rd(P.OFF_ALGO_LO)),
                       "algo_stages": P.algo_stages((self._rd(P.OFF_ALGO_HI) << 32) | self._rd(P.OFF_ALGO_LO)),
                       "status": hx(status), "result": hx(self._rd(P.OFF_ISP_RESULT)),
                       "epoch": self._rd(P.OFF_ISP_RESULT) & 0xFFFF,
                       "source": (self._rd(P.OFF_ISP_RESULT) >> 16) & 3}
                self._record("ISP", "QUERY", [], res)
                return res
            except Exception as exc:
                return {"ok": False, "state": "DISCONNECTED", "error": str(exc)}

    def isp_apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not self.available:
                return self._unavailable()
            try:
                tiles = payload.get("tiles")
                if tiles is None:
                    if "tile_fields" in payload:
                        tiles = [P.isp_tile_dwords(P.isp_tile_encode(x)) for x in payload["tile_fields"]]
                    else:
                        tiles = [list(payload.get("tile", P.ISP_RESET_DW)) for _ in range(4)]
                if len(tiles) != 4:
                    raise ValueError("four tiles required")
                normalized = []
                for t in tiles:
                    if isinstance(t, dict):
                        normalized.append(P.isp_tile_dwords(P.isp_tile_encode(t)))
                    elif len(t) == 4:
                        normalized.append([u32(x) for x in t])
                    else:
                        raise ValueError("each tile needs 4 DWORDs or field map")
                global_word = u32(payload.get("global", P.ISP_GLOBAL_RESET))
            except Exception as exc:
                return {"ok": False, "state": "REJECTED", "error": str(exc)}
            writes = [(P.OFF_ISP_GLOBAL, global_word)]
            writes += [(P.OFF_ISP_TILE + i * 16 + j * 4, v) for i, t in enumerate(normalized) for j, v in enumerate(t)]
            if self.sim:
                self.image.global_word = global_word
                self.image.tiles = normalized
                self.image.epoch = (self.image.epoch + 1) & 0xFFFF
                self.image.source = 2
                res = {"ok": True, "state": "APPLIED", "sequence": self.image.epoch,
                       "global": hx(global_word), "tiles": [[hx(x) for x in t] for t in normalized],
                       "tile_fields": [P.isp_tile_decode(P.isp_words_to_value(t)) for t in normalized],
                       "epoch": self.image.epoch, "source": self.image.source,
                       "status": hx(self._image_status(self.image.epoch))}
                self._record("ISP", "APPLY", writes, res)
                return res
            try:
                self._wr(P.OFF_ISP_GLOBAL, global_word)
                for i, t in enumerate(normalized):
                    for j, v in enumerate(t):
                        self._wr(P.OFF_ISP_TILE + i * 16 + j * 4, v)
                res = self._image_tx("ISP", P.OFF_ISP_COMMIT, P.OFF_ISP_STATUS, P.OFF_ISP_RESULT,
                                     P.OFF_ISP_CLEAR, 1)
                res.update({"global": hx(global_word), "tiles": [[hx(x) for x in t] for t in normalized],
                            "tile_fields": [P.isp_tile_decode(P.isp_words_to_value(t)) for t in normalized]})
                self._record("ISP", "APPLY", writes, res)
                return res
            except Exception as exc:
                return {"ok": False, "state": "DISCONNECTED", "error": str(exc)}

    def _image_tx(self, label: str, commit: int, status_off: int, result_off: int,
                  clear_off: int, opcode: int) -> dict[str, Any]:
        st = self._rd(status_off)
        if st & 1:
            return {"ok": False, "state": "TIMEOUT_UNCONFIRMED", "error": "busy before commit", "status": hx(st)}
        last = (self._rd(commit) >> 16) & 0xFFFF
        seq = (last + 1) & 0xFFFF
        self._wr(clear_off, 1)
        self._wr(commit, (seq << 16) | opcode)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            st = self._rd(status_off)
            if st & 4:
                return {"ok": False, "state": "REJECTED", "sequence": seq,
                        "status": hx(st), "error": (st >> 8) & 0xFF}
            if not (st & 1) and (st & 2) and ((st >> 16) & 0xFFFF) == seq:
                result = self._rd(result_off)
                return {"ok": True, "state": "APPLIED", "sequence": seq, "status": hx(st),
                        "epoch": result & 0xFFFF, "source": (result >> 16) & 3,
                        "result": hx(result)}
            time.sleep(0.002)
        return {"ok": False, "state": "TIMEOUT_UNCONFIRMED", "sequence": seq,
                "status": hx(self._rd(status_off))}

    def algo_query(self) -> dict[str, Any]:
        with self.lock:
            if not self.available:
                return self._unavailable()
            if self.sim:
                self.image.epoch = (self.image.epoch + 1) & 0xFFFF
                self.image.source = 2
                res = {"ok": True, "state": "READ", "algo": hx(self.image.algo, 16),
                       "fields": P.algo_decode(self.image.algo), "stages": P.algo_stages(self.image.algo),
                       "epoch": self.image.epoch, "source": self.image.source}
                self._record("ALGO", "QUERY", [], res)
                return res
            try:
                res = self._image_tx("ALGO", P.OFF_ALGO_COMMIT, P.OFF_ALGO_STATUS,
                                     P.OFF_ALGO_RESULT, P.OFF_ALGO_CLEAR, 2)
                value = (self._rd(P.OFF_ALGO_SNAP_HI) << 32) | self._rd(P.OFF_ALGO_SNAP_LO)
                res.update({"algo": hx(value, 16), "fields": P.algo_decode(value),
                            "stages": P.algo_stages(value)})
                self._record("ALGO", "QUERY", [], res)
                return res
            except Exception as exc:
                return {"ok": False, "state": "DISCONNECTED", "error": str(exc)}

    def algo_apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not self.available:
                return self._unavailable()
            try:
                if "fields" in payload:
                    value = P.algo_encode(payload["fields"])
                elif "value" in payload:
                    value = P.u64(int(str(payload["value"]), 0))
                else:
                    value = self.image.algo
                for stage, enable in payload.get("stages", {}).items():
                    value = P.algo_with_stage(value, stage, bool(enable))
                if value >> 60:
                    raise ValueError("reserved ALGO bits 63:60 must be zero")
                if (value >> 52) & 0xF:
                    raise ValueError("reserved ALGO bits 55:52 must be zero")
                if (value >> 17) & 0x7F:
                    raise ValueError("reserved ALGO bits 23:17 must be zero")
            except Exception as exc:
                return {"ok": False, "state": "REJECTED", "error": str(exc)}
            writes = [(P.OFF_ALGO_LO, value & 0xFFFFFFFF), (P.OFF_ALGO_HI, value >> 32)]
            if self.sim:
                self.image.algo = value
                self.image.epoch = (self.image.epoch + 1) & 0xFFFF
                self.image.source = 2
                res = {"ok": True, "state": "APPLIED", "algo": hx(value, 16),
                       "fields": P.algo_decode(value), "stages": P.algo_stages(value),
                       "epoch": self.image.epoch, "source": self.image.source}
                self._record("ALGO", "APPLY", writes, res)
                return res
            try:
                self._wr(P.OFF_ALGO_LO, value & 0xFFFFFFFF)
                self._wr(P.OFF_ALGO_HI, value >> 32)
                res = self._image_tx("ALGO", P.OFF_ALGO_COMMIT, P.OFF_ALGO_STATUS,
                                     P.OFF_ALGO_RESULT, P.OFF_ALGO_CLEAR, 1)
                res.update({"algo": hx(value, 16), "fields": P.algo_decode(value),
                            "stages": P.algo_stages(value)})
                self._record("ALGO", "APPLY", writes, res)
                return res
            except Exception as exc:
                return {"ok": False, "state": "DISCONNECTED", "error": str(exc)}

    # ---------------------------------------------------------------- diagnostics
    def diag(self) -> dict[str, Any]:
        with self.lock:
            if not self.available:
                return self._unavailable()
            if self.sim:
                behavior = P.behavior_encode({"throttle_allow": True, "brake_allow": True,
                                              "left_allow": True, "right_allow": False,
                                              "right_forbid": True, "reason": 2, "action_active": True})
                hsst = {"sequence": 42, "payload": hx(self.hsst_payload),
                        "status": hx(self.hsst_status), "protocol_id": "0x4c4d0001",
                        "decoded": P.hsst_decode(self.hsst_payload)}
                return {"ok": True, "behavior": P.behavior_decode(behavior),
                        "behavior_raw": hx(behavior), "hsst": hsst,
                        "video_dma": {"available": False, "state": "VIDEO_DMA_CONTROL_UNAVAILABLE"},
                        "legacy_dma": {"available": False, "state": "DMA_CONTROL_UNAVAILABLE"}}
            try:
                behavior = self._rd(P.OFF_DIAG)
                hsst = {"sequence": self._rd(P.OFF_HSST_SEQ), "payload": hx(self._rd(P.OFF_HSST_PAYLOAD)),
                        "status": hx(self._rd(P.OFF_HSST_STATUS)), "protocol_id": hx(self._rd(P.OFF_HSST_PROTOCOL)),
                        "decoded": P.hsst_decode(self._rd(P.OFF_HSST_PAYLOAD))}
                return {"ok": True, "behavior": P.behavior_decode(behavior), "behavior_raw": hx(behavior),
                        "hsst": hsst,
                        "video_dma": {"available": False, "state": "VIDEO_DMA_CONTROL_UNAVAILABLE"},
                        "legacy_dma": {"available": False, "state": "DMA_CONTROL_UNAVAILABLE"}}
            except Exception as exc:
                return {"ok": False, "state": "DISCONNECTED", "error": str(exc)}

    def read_registers(self, start: int = 0x600, count: int = 64) -> dict[str, Any]:
        with self.lock:
            if not self.available:
                return self._unavailable()
            if start % 4 or start < 0x100 or start > 0x7FC:
                return {"ok": False, "error": "offset must be 4-aligned in BAR1 control window"}
            count = max(1, min(128, int(count)))
            out = []
            for i in range(count):
                off = start + i * 4
                if off > 0x7FC:
                    break
                value = self._rd(off) if not self.sim else 0
                out.append({"offset": hx(off), "value": hx(value)})
            return {"ok": True, "registers": out}

    def command(self, req: dict[str, Any]) -> dict[str, Any]:
        group = str(req.get("group", "")).lower()
        action = str(req.get("action", "")).lower()
        try:
            if group == "hud":
                return self.hud_read() if action in ("read", "status") else self.hud_apply(req)
            if group == "scene":
                return self.scene_read() if action in ("read", "status") else self.scene_apply(req)
            if group == "nav":
                if action in ("read", "status"):
                    return self.nav_read()
                if action == "clear_errors":
                    if not self.available:
                        return self._unavailable()
                    if self.sim:
                        self.nav.status = 0x0E
                        return {"ok": True, "state": "CLEARED", "status": "0x0000000e"}
                    self._wr(P.OFF_NAV_CONTROL, 2)
                    return {"ok": True, "state": "CLEARED", "status": hx(self._rd(P.OFF_NAV_STATUS))}
                return self.nav_apply(req)
            if group == "isp":
                return self.isp_query() if action in ("read", "query") else self.isp_apply(req)
            if group == "algo":
                return self.algo_query() if action in ("read", "query") else self.algo_apply(req)
            if group in ("diag", "behavior", "hsst"):
                return self.diag()
            if group == "registers":
                return self.read_registers(int(str(req.get("start", "0x600")), 0), int(req.get("count", 64)))
            return {"ok": False, "state": "REJECTED", "error": f"unknown group {group}"}
        except Exception as exc:
            return {"ok": False, "state": "REJECTED", "error": f"{type(exc).__name__}: {exc}"}

    def golden(self, name: str) -> dict[str, Any]:
        if name == "hud_left_adas":
            return self.hud_apply({**{k: False for k, _, _ in P.HUD_BITS}, "hud_enable": True,
                                   "left": True, "adas": True, "sequence": 1})
        if name == "hud_release":
            return self.hud_apply({**{k: False for k, _, _ in P.HUD_BITS}, "sequence": 2,
                                   "release_takeover": True})
        if name == "scene":
            return self.scene_apply({"valid": True, "traffic": 2, "person": False, "obstacle": False,
                                     "speed_limit": False, "left_lane": 2, "right_lane": 1,
                                     "warning": False, "show_time": True,
                                     "time0": "0x10242026", "time1": "0x00053014", "sequence": 42})
        if name == "nav":
            return self.nav_apply({"enabled": True, "direction": "left", "distance_m": 120, "sequence": 1})
        return {"ok": False, "error": "unknown golden"}


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    service: Service
    token = ""

    def log_message(self, fmt, *args):
        return

    def _authorized(self) -> bool:
        if not self.token:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {self.token}"

    def _send(self, code: int, data: Any, content_type="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, (HERE / "pcie_web" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/app.js":
            return self._send(200, (HERE / "pcie_web" / "app.js").read_bytes(), "application/javascript")
        if path == "/style.css":
            return self._send(200, (HERE / "pcie_web" / "style.css").read_bytes(), "text/css")
        if not self._authorized():
            return self._send(401, {"ok": False, "error": "unauthorized"})
        if path == "/api/status":
            return self._send(200, self.service.status())
        if path == "/api/log":
            return self._send(200, {"ok": True, "logs": self.service.logs[-50:]})
        if path == "/api/registers":
            q = parse_qs(urlparse(self.path).query)
            return self._send(200, self.service.read_registers(int(q.get("start", ["0x600"])[0], 0), int(q.get("count", ["64"])[0])))
        return self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if not self._authorized():
            return self._send(401, {"ok": False, "error": "unauthorized"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length) or b"{}")
            path = urlparse(self.path).path
            if path == "/api/command":
                return self._send(200, self.service.command(req))
            if path.startswith("/api/golden/"):
                return self._send(200, self.service.golden(path.rsplit("/", 1)[-1]))
            return self._send(404, {"ok": False, "error": "not found"})
        except Exception as exc:
            return self._send(400, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def udp_loop(service: Service, port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    while True:
        ready, _, _ = select.select([sock], [], [], 1.0)
        if not ready:
            continue
        data, peer = sock.recvfrom(65535)
        try:
            req = json.loads(data)
            typ = req.get("type", "command")
            if typ == "caps":
                reply = {"type": "caps", **service.status()}
            elif typ == "query":
                reply = {"type": "query", **service.isp_query()}
            elif typ == "apply_isp":
                reply = {"type": "apply_isp", **service.isp_apply(req)}
            elif typ == "apply_algo":
                reply = {"type": "apply_algo", **service.algo_apply(req)}
            else:
                reply = {"type": typ, **service.command(req)}
        except Exception as exc:
            reply = {"ok": False, "state": "REJECTED", "error": f"{type(exc).__name__}: {exc}"}
        sock.sendto(json.dumps(reply, ensure_ascii=False).encode(), peer)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bar1", default="/sys/bus/pci/devices/0000:01:00.0/resource1")
    parser.add_argument("--port", type=int, default=38264)
    parser.add_argument("--web-port", type=int, default=8088)
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--token", default="")
    parser.add_argument("--simulation", action="store_true")
    args = parser.parse_args()
    service = Service(args.bar1, simulation=args.simulation)
    Handler.service = service
    Handler.token = args.token
    httpd = ThreadingHTTPServer((args.web_host, args.web_port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"READY mode={service.mode} available={service.available} error={service.error} "
          f"web=http://{args.web_host}:{args.web_port} udp={args.port}", flush=True)
    try:
        udp_loop(service, args.port)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
