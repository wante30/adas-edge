#!/usr/bin/env python3
"""Root-owned PCIe BAR1 HUD V2 bridge for the Qt test UI.

Qt never opens resource1 directly. It sends validated JSON snapshots to
127.0.0.1:38262; this helper maps BAR1 and performs the ordered V2 mailbox
transaction plus the legacy HUD enable word.

Run on the board:
    sudo python3 hud_bridge.py
"""
from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import socket
import time
from pathlib import Path

MAGIC = 0x31445548
OFF_MAGIC = 0x600
OFF_VERSION = 0x604
OFF_REQUEST = 0x608
OFF_ACCEPTED = 0x610
OFF_APPLIED = 0x614
OFF_RK_STATE = 0x618
OFF_EFFECTIVE = 0x61C
OFF_REJECT = 0x620
OFF_ERROR = 0x624
OFF_EXT_REQUEST = 0x638
OFF_TIME0 = 0x63C
OFF_TIME1 = 0x640
OFF_EXT_COMMIT = 0x644
OFF_EXT_STATUS = 0x648
OFF_EXT_APPLIED = 0x64C


class Bridge:
    def __init__(self, resource: Path):
        self.fd = os.open(str(resource), os.O_RDWR | getattr(os, "O_SYNC", 0))
        self.mm = mmap.mmap(self.fd, 4096, access=mmap.ACCESS_WRITE)
        self.ext_seq = 0
        self.legacy_seq = 0
        if self.rd(OFF_MAGIC) != MAGIC:
            raise SystemExit("HUD magic mismatch; refuse to write BAR1")

    def rd(self, offset: int) -> int:
        return ctypes.c_uint32.from_buffer(self.mm, offset).value

    def wr(self, offset: int, value: int) -> None:
        ctypes.c_uint32.from_buffer(self.mm, offset).value = int(value) & 0xFFFFFFFF

    def status(self) -> dict:
        return {
            "type": "status",
            "ok": True,
            "magic": self.rd(OFF_MAGIC),
            "version": self.rd(OFF_VERSION),
            "caps": self.rd(OFF_MAGIC + 0x30),
            "ext_status": self.rd(OFF_EXT_STATUS),
            "ext_applied": self.rd(OFF_EXT_APPLIED),
            "reject_count": self.rd(OFF_REJECT),
            "error": self.rd(OFF_ERROR),
        }

    def _wait_ready(self, timeout_s: float = 0.15) -> int:
        deadline = time.monotonic() + timeout_s
        status = self.rd(OFF_EXT_STATUS)
        while time.monotonic() < deadline:
            status = self.rd(OFF_EXT_STATUS)
            if not (status & 1) and (status & 2):
                return status
            time.sleep(0.001)
        return status

    def _legacy(self, state: int, blink: int) -> dict:
        self.legacy_seq = self.legacy_seq % 0xFFFF + 1
        word = (self.legacy_seq << 16) | (1 << 12) | ((blink & 7) << 9) | (state & 0x1FF)
        self.wr(OFF_REQUEST, word)
        # Read the accepted/applied sequence; this is a status check, not a
        # claim that the HDMI pixel boundary has already consumed it.
        accepted = self.rd(OFF_ACCEPTED)
        applied = self.rd(OFF_APPLIED)
        return {
            "legacy_seq": self.legacy_seq,
            "legacy_word": word,
            "legacy_accepted": accepted,
            "legacy_applied": applied,
            "legacy_rk_state": self.rd(OFF_RK_STATE),
            "legacy_effective": self.rd(OFF_EFFECTIVE),
        }

    def set(self, request: dict) -> dict:
        def flag(name: str) -> bool:
            return bool(request.get(name, False))

        def int_field(name: str, default: int = 0) -> int:
            value = request.get(name, default)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be an int")
            value = int(value)
            if not 0 <= value <= 2:
                raise ValueError(f"{name} out of range")
            return value

        traffic = int_field("traffic", 0)
        lane_left = int_field("lane_left", 0)
        lane_right = int_field("lane_right", 0)
        ext = 0
        if flag("perception_enable"):
            ext |= 1
        ext |= traffic << 1
        if flag("pedestrian"):
            ext |= 1 << 3
        if flag("static_target"):
            ext |= 1 << 4
        if flag("overspeed"):
            ext |= 1 << 5
        ext |= lane_left << 6
        ext |= lane_right << 8
        if flag("lka_warning"):
            ext |= 1 << 10

        time_enable = flag("time_enable")
        time0 = int(request.get("time0", 0)) & 0xFFFFFFFF
        time1 = int(request.get("time1", 0)) & 0xFFFFFFFF
        if time_enable:
            ext |= 1 << 11
        else:
            time0 = 0
            time1 = 0

        status = self._wait_ready()
        if status & 1:
            return {"ok": False, "reason": "mailbox pending", "ext_status": status}

        self.ext_seq = self.ext_seq % 0xFFFF + 1
        self.wr(OFF_EXT_REQUEST, ext)
        self.wr(OFF_TIME0, time0)
        self.wr(OFF_TIME1, time1)
        self.wr(OFF_EXT_COMMIT, self.ext_seq)

        deadline = time.monotonic() + 0.15
        while time.monotonic() < deadline:
            status = self.rd(OFF_EXT_STATUS)
            if not (status & 1) and (status & 2):
                break
            time.sleep(0.001)
        applied = self.rd(OFF_EXT_APPLIED)
        result = {
            "type": "ack",
            "ok": bool(not (status & 1) and applied == self.ext_seq),
            "ext_request": ext,
            "ext_seq": self.ext_seq,
            "ext_applied": applied,
            "ext_status": status,
            "error": self.rd(OFF_ERROR),
            "reject_count": self.rd(OFF_REJECT),
            "time0": time0,
            "time1": time1,
        }
        if flag("legacy_enable"):
            state = int(request.get("legacy_state", 0x101)) & 0x1FF
            result.update(self._legacy(state, 0))
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bar1",
        type=Path,
        default=Path("/sys/bus/pci/devices/0000:01:00.0/resource1"),
    )
    parser.add_argument("--port", type=int, default=38262)
    args = parser.parse_args()

    bridge = Bridge(args.bar1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    sock.settimeout(0.5)
    print(
        f"READY bar1={args.bar1} port={args.port} "
        f"magic=0x{bridge.rd(OFF_MAGIC):08x} caps=0x{bridge.rd(OFF_MAGIC + 0x30):08x}",
        flush=True,
    )
    while True:
        try:
            packet, peer = sock.recvfrom(4096)
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            request = json.loads(packet.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("JSON object required")
            if request.get("type") == "status":
                reply = bridge.status()
            elif request.get("type") == "set":
                reply = bridge.set(request)
            else:
                reply = {"ok": False, "reason": "unknown command"}
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            reply = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        except OSError as exc:
            reply = {"ok": False, "reason": f"BAR1 error: {exc}"}
        try:
            sock.sendto(json.dumps(reply, separators=(",", ":")).encode("utf-8"), peer)
        except OSError:
            pass
        if not reply.get("ok", False):
            print(f"REJECT {reply}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
