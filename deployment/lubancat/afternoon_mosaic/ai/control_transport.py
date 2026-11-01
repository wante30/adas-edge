"""Replay acknowledgements without creating a socket or touching a control port."""
import json
import time
import uuid


class DryRunSocket:
    """Datagram-shaped adapter for the preserved hardware controller interface."""
    def __init__(self):
        self.pending = None

    def settimeout(self, seconds):
        pass

    def sendto(self, payload, address):
        self.pending = json.loads(payload.decode("utf-8"))
        return len(payload)

    def recvfrom(self, size):
        if self.pending is None:
            raise OSError("No dry-run request")
        request, self.pending = self.pending, None
        reply = {"ok": True, "accepted": True, "executed": False,
                 "simulated": True, "control_mode": "DRY_RUN",
                 "request_id": request.get("request_id") or uuid.uuid4().hex,
                 "timestamp_ns": time.monotonic_ns(), "last_error": 0}
        print("CONTROL_DRY_RUN " + json.dumps({"request": request, "ack": reply},
              separators=(",", ":")), flush=True)
        return json.dumps(reply).encode("utf-8"), ("DRY_RUN", 0)

    def close(self):
        self.pending = None
