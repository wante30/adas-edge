"""Record test brake requests without accessing the HUD bridge or PCIe."""
import json
from pathlib import Path
import socket
import sys
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", int(sys.argv[1])))
with Path(sys.argv[2]).open("a", encoding="utf-8", buffering=1) as log:
    while True:
        payload, peer = sock.recvfrom(8192)
        request = json.loads(payload)
        ok = request.get("type") == "action" and request.get("action") == "brake"
        log.write(json.dumps({"monotonic_ns": time.monotonic_ns(), "request": request,
                              "simulated": True, "ok": ok}) + "\n")
        sock.sendto(json.dumps({"ok": ok, "simulated": True, "last_error": 0}).encode(), peer)
