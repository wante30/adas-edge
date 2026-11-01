import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

AI = Path(__file__).parents[1] / "ai"
sys.path.insert(0, str(AI))
from auto_brake import AutoBrakeController


class DryRunTests(unittest.TestCase):
    def test_replay_has_no_socket_or_udp_io(self):
        with patch.dict(os.environ, {"ADAS_CONTROL_MODE": "DRY_RUN"}), patch("socket.socket", side_effect=AssertionError("network forbidden")):
            controller = AutoBrakeController(enabled=True)
            payload = json.dumps({"action": "brake", "request_id": "test-1"}).encode()
            controller.sock.sendto(payload, ("127.0.0.1", 38262))
            reply, _ = controller.sock.recvfrom(4096)
            ack = json.loads(reply)
            self.assertTrue(ack["accepted"])
            self.assertTrue(ack["simulated"])
            self.assertFalse(ack["executed"])
            self.assertEqual(ack["request_id"], "test-1")
