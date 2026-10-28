#!/usr/bin/env python3
"""Send one V2 perception snapshot to the local root bridge and print its reply."""
from __future__ import annotations

import json
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(1.0)
sock.sendto(json.dumps({"type": "status"}).encode(), ("127.0.0.1", 38262))
try:
    print("status", sock.recvfrom(4096)[0].decode())
except socket.timeout:
    print("status timeout; bridge not running")
sock.sendto(json.dumps({
    "type": "set",
    "perception_enable": True,
    "traffic": 1,
    "pedestrian": True,
    "static_target": False,
    "overspeed": False,
    "lane_left": 2,
    "lane_right": 1,
    "lka_warning": False,
    "time_enable": False,
    "legacy_enable": True,
    "legacy_state": 257,
}).encode(), ("127.0.0.1", 38262))
try:
    print("set", sock.recvfrom(4096)[0].decode())
except socket.timeout:
    print("set timeout")
