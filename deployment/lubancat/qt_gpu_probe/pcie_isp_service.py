#!/usr/bin/env python3
"""ISP/ALGO control service, with a simulation backend when BAR1 is absent.

The service never writes video DMA/unknown registers.  Hardware writes are
limited to the documented ISP/ALGO shadow, commit and clear windows.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import select
import socket
import time
from dataclasses import dataclass, field

MAGIC = 0x31435049
ISP_GLOBAL = 0x70C
ISP_TILE = 0x710
ISP_COMMIT = 0x750
ISP_STATUS = 0x754
ISP_RESULT = 0x758
ISP_CLEAR = 0x75C
ALGO_LO = 0x760
ALGO_HI = 0x764
ALGO_COMMIT = 0x768
ALGO_STATUS = 0x76C
ALGO_RESULT = 0x770
ALGO_CLEAR = 0x774
SNAP_INDEX = 0x780
SNAP_DATA = 0x784
ALGO_SNAP_LO = 0x788
ALGO_SNAP_HI = 0x78C
TILE_RESET = [0x40000000, 0x21840410, 0x00000001, 0x00401004]
ALGO_RESET = (0x000FE619 << 32) | 0xDC010080


def u32(v: int) -> int:
    return int(v) & 0xFFFFFFFF


@dataclass
class State:
    global_shadow: int = 0x1E
    tiles: list = field(default_factory=lambda: [list(TILE_RESET) for _ in range(4)])
    algo: int = ALGO_RESET
    epoch: int = 0
    source: int = 0
    applied: bool = False

    def snapshot(self) -> dict:
        return {
            "global": f"0x{self.global_shadow:08x}",
            "tiles": [[f"0x{x:08x}" for x in t] for t in self.tiles],
            "algo": f"0x{self.algo:016x}",
            "epoch": self.epoch,
            "source": self.source,
            "applied": self.applied,
        }


class Backend:
    def __init__(self, resource: str, allow_simulation: bool = False):
        self.resource = resource
        self.fd = None
        self.mm = None
        self.sim = False
        self.available = False
        self.mode = "DISCONNECTED"
        self.error = ""
        self.simstate = State()
        self._allow_simulation = bool(allow_simulation)
        if self._allow_simulation:
            self.sim = True
            self.available = True
            self.mode = "SIMULATION"
            self.error = "explicit simulation mode"
        else:
            self._open()

    def _open(self) -> None:
        try:
            self.fd = os.open(self.resource, os.O_RDWR | getattr(os, "O_SYNC", 0))
            import mmap
            self.mm = mmap.mmap(self.fd, 4096, access=mmap.ACCESS_WRITE)
            if self.rd(0x700) == MAGIC:
                self.sim = False
                self.available = True
                self.mode = "HARDWARE"
            else:
                self.available = False
                self.mode = "DISCONNECTED"
                self.error = "BAR1 magic mismatch"
        except Exception as exc:
            self.available = False
            self.mode = "DISCONNECTED"
            self.error = f"{type(exc).__name__}: {exc}"

    def rd(self, off: int) -> int:
        if self.mm is None:
            raise RuntimeError("BAR1 not mapped")
        return ctypes.c_uint32.from_buffer(self.mm, off).value

    def wr(self, off: int, val: int) -> None:
        if self.mm is None:
            raise RuntimeError("BAR1 not mapped")
        ctypes.c_uint32.from_buffer(self.mm, off).value = u32(val)

    def caps(self) -> dict:
        if self.sim:
            return {"mode": self.mode, "available": True, "error": self.error,
                    "magic": f"0x{MAGIC:08x}", "version": 1, "caps": "0x0000000f"}
        if not self.available:
            return {"mode": self.mode, "available": False, "error": self.error}
        return {"mode": self.mode, "available": True, "error": "",
                "magic": f"0x{self.rd(0x700):08x}", "version": self.rd(0x704),
                "caps": f"0x{self.rd(0x708):08x}"}

    def _wait_idle(self, status_off: int, timeout=1.0) -> int:
        end = time.monotonic() + timeout
        st = self.rd(status_off)
        while time.monotonic() < end:
            st = self.rd(status_off)
            if not (st & 1):
                return st
            time.sleep(0.001)
        return st

    def _transaction(self, channel: str, opcode: int, timeout=3.0) -> dict:
        if self.sim:
            # Simulation applies immediately but preserves the real status vocabulary.
            self.simstate.epoch = (self.simstate.epoch + 1) & 0xFFFF
            self.simstate.source = 2
            self.simstate.applied = True
            return {"ok": True, "state": "applied", "epoch": self.simstate.epoch}
        commit, status, clear = ((ISP_COMMIT, ISP_STATUS, ISP_CLEAR)
                                 if channel == "isp" else (ALGO_COMMIT, ALGO_STATUS, ALGO_CLEAR))
        st = self._wait_idle(status)
        if st & 1:
            return {"ok": False, "state": "timeout_unconfirmed", "status": st}
        try:
            last = (self.rd(commit) >> 16) & 0xFFFF
        except Exception:
            last = 0
        seq = (last + 1) & 0xFFFF
        self.wr(clear, 1)
        self.wr(commit, (seq << 16) | opcode)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            st = self.rd(status)
            if st & 4:
                return {"ok": False, "state": "rejected", "sequence": seq,
                        "status": st, "error": (st >> 8) & 0xFF}
            if not (st & 1) and (st & 2) and ((st >> 16) & 0xFFFF) == seq:
                res = self.rd(ISP_RESULT if channel == "isp" else ALGO_RESULT)
                return {"ok": True, "state": "applied", "sequence": seq,
                        "status": st, "epoch": res & 0xFFFF,
                        "source": (res >> 16) & 0x3}
            time.sleep(0.001)
        return {"ok": False, "state": "timeout_unconfirmed", "sequence": seq,
                "status": self.rd(status)}

    def _unavailable(self) -> dict:
        return {"ok": False, "state": "disconnected", "mode": self.mode,
                "error": self.error or "BAR1 unavailable"}

    def query(self) -> dict:
        if not self.available:
            return self._unavailable()
        if self.sim:
            return {"ok": True, "state": "applied", **self.simstate.snapshot()}
        # QUERY is a write transaction followed by shadow/snapshot reads.
        tx = self._transaction("isp", 2)
        tx["global"] = f"0x{self.rd(ISP_GLOBAL):08x}"
        tx["tiles"] = [[f"0x{self.rd(ISP_TILE + i * 16 + j * 4):08x}" for j in range(4)]
                       for i in range(4)]
        tx["algo"] = f"0x{(self.rd(ALGO_HI) << 32) | self.rd(ALGO_LO):016x}"
        return tx

    def apply_isp(self, payload: dict) -> dict:
        if not self.available:
            return self._unavailable()
        global_value = int(str(payload.get("global", "0x1e")), 0) & 0xFF
        tiles = payload.get("tiles")
        if not isinstance(tiles, list) or len(tiles) != 4:
            return {"ok": False, "state": "rejected", "error": "four tile arrays required"}
        normalized = []
        for tile in tiles:
            if not isinstance(tile, list) or len(tile) != 4:
                return {"ok": False, "state": "rejected", "error": "tile needs 4 dwords"}
            normalized.append([int(str(x), 0) & 0xFFFFFFFF for x in tile])
        if self.sim:
            self.simstate.global_shadow = global_value
            self.simstate.tiles = normalized
            return {"ok": True, "state": "applied", **self.simstate.snapshot()}
        # Write shadow only after validation, then APPLY and QUERY back.
        self.wr(ISP_GLOBAL, global_value)
        for i, tile in enumerate(normalized):
            for j, value in enumerate(tile):
                self.wr(ISP_TILE + i * 16 + j * 4, value)
        tx = self._transaction("isp", 1)
        if tx.get("ok"):
            tx.update(self.query())
        return tx

    def apply_algo(self, payload: dict) -> dict:
        if not self.available:
            return self._unavailable()
        value = int(str(payload.get("value", hex(ALGO_RESET))), 0) & ((1 << 64) - 1)
        if self.sim:
            self.simstate.algo = value
            return {"ok": True, "state": "applied", "algo": f"0x{value:016x}"}
        self.wr(ALGO_LO, value & 0xFFFFFFFF)
        self.wr(ALGO_HI, value >> 32)
        return self._transaction("algo", 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bar1", default="/sys/bus/pci/devices/0000:01:00.0/resource1")
    parser.add_argument("--port", type=int, default=38264)
    parser.add_argument("--simulation", action="store_true",
                        help="explicitly enable simulated ISP/ALGO backend")
    args = parser.parse_args()
    backend = Backend(args.bar1, allow_simulation=args.simulation)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    print(f"READY mode={backend.mode} available={backend.available} error={backend.error}", flush=True)
    while True:
        ready, _, _ = select.select([sock], [], [], 1.0)
        if not ready:
            continue
        data, peer = sock.recvfrom(65535)
        try:
            req = json.loads(data)
            typ = req.get("type")
            if typ == "caps":
                reply = {"type": "caps", "ok": True, **backend.caps()}
            elif typ == "query":
                reply = {"type": "query", **backend.query()}
            elif typ == "apply_isp":
                reply = {"type": "apply_isp", **backend.apply_isp(req)}
            elif typ == "apply_algo":
                reply = {"type": "apply_algo", **backend.apply_algo(req)}
            else:
                reply = {"type": "error", "ok": False, "error": "unknown request"}
        except Exception as exc:
            reply = {"type": "error", "ok": False, "state": "rejected",
                     "error": f"{type(exc).__name__}: {exc}"}
        reply.setdefault("mode", backend.mode)
        reply.setdefault("available", backend.available)
        sock.sendto(json.dumps(reply).encode(), peer)


if __name__ == "__main__":
    raise SystemExit(main())
