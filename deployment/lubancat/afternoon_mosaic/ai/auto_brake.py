"""Automatic brake request sender for hazard events.

The root HUD bridge owns BAR1.  This controller only sends validated brake
requests to that local bridge; it never accesses PCIe directly and never sends
throttle.
"""
from __future__ import annotations

import json
import os
import socket
import time

BRAKE_KINDS = {
    "NEAR_OBJECT_IN_PATH",
    "FRONT_COLLISION_RISK",
    "PEDESTRIAN_IN_PATH",
    "PEDESTRIAN_CROSSING_PATH",
}
SEVERITIES = {"warning", "critical"}


class AutoBrakeController:
    def __init__(self, *, enabled=None, port=38262):
        if enabled is None:
            enabled = os.environ.get("ADAS_AUTO_BRAKE") == "1"
        self.enabled = bool(enabled)
        self.address = ("127.0.0.1", int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.08)
        self.last_key = None
        self.blocked_key = None
        self.last_send_s = 0.0
        self.last_result = None

    @staticmethod
    def _key(events):
        return tuple(sorted(
            (str(event.get("kind")), int(event.get("track_id") or 0))
            for event in events
        ))

    @staticmethod
    def _active(events):
        return tuple(
            event for event in events
            if event.get("kind") in BRAKE_KINDS
            and event.get("severity") in SEVERITIES
        )

    def update(self, hazards):
        if not self.enabled:
            return None
        events = getattr(hazards, "events", ()) or ()
        active = self._active(events)
        if not active:
            self.last_key = None
            self.blocked_key = None
            return None
        key = self._key(active)
        if key != self.last_key:
            self.last_key = key
            self.blocked_key = None
            self.last_send_s = 0.0
        if key == self.blocked_key:
            return None
        now = time.monotonic()
        if now - self.last_send_s < 0.4:
            return None
        self.last_send_s = now
        request = {
            "type": "action",
            "action": "brake",
            "source": "hazard",
            "events": [
                {
                    "kind": event.get("kind"),
                    "severity": event.get("severity"),
                    "track_id": event.get("track_id"),
                }
                for event in active
            ],
        }
        try:
            self.sock.sendto(
                json.dumps(request, separators=(",", ":")).encode("utf-8"),
                self.address,
            )
            reply, _ = self.sock.recvfrom(4096)
            result = json.loads(reply.decode("utf-8"))
            self.last_result = result
            print(f"AUTO_BRAKE {result}", flush=True)
            if not result.get("ok", False):
                if result.get("last_error") == 6:
                    self.blocked_key = key
                else:
                    self.blocked_key = key
            return result
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.last_result = {"ok": False, "reason": type(exc).__name__}
            print(f"AUTO_BRAKE_FAILED {type(exc).__name__}: {exc}", flush=True)
            self.blocked_key = key
            return self.last_result
