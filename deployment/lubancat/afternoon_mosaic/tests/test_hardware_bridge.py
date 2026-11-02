import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hardware"))
from hardware_bridge import dispatch, publish_vehicle
from vehicle_telemetry_v1 import MAGIC, VERSION, REG

class HardwareBridgeTest(unittest.TestCase):
    def test_real_ack_keeps_word_and_correlates_request_without_claiming_execution(self):
        bridge = Mock()
        word = (23 << 16) | 2
        bridge.action.return_value = dict(ok=True, accepted=word, request=word,
            sequence=23, status=(23 << 16) | (2 << 5) | 2 | 8)
        result = dispatch(bridge, dict(type="action", action="brake", request_id="critical:4"))
        self.assertEqual(result["request_id"], "critical:4")
        self.assertEqual(result["accepted"], word)
        self.assertTrue(result["fpga_accepted"])
        self.assertTrue(result["fpga_action_active"])
        self.assertFalse(result["executed"])
        self.assertFalse(result["simulated"])

    def test_rejected_or_wrong_sequence_cannot_claim_active(self):
        bridge = Mock()
        bridge.action.return_value = dict(ok=False, accepted=12, request=15, sequence=2, status=2)
        self.assertFalse(dispatch(bridge, dict(type="action", action="brake"))["fpga_accepted"])
        bridge.action.return_value = dict(ok=True, accepted=15, request=15, sequence=2,
            status=(3 << 16) | (2 << 5) | 2)
        self.assertFalse(dispatch(bridge, dict(type="action", action="brake"))["fpga_action_active"])

    def test_incompatible_mmio_publishes_offline_instead_of_fresh_online(self):
        bridge = Mock(); bridge.rd.return_value = 0xffffffff
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"status.json"
            value = publish_vehicle(bridge, target)
            self.assertFalse(value["online"])
            self.assertTrue(value["sensor_stale"])
            self.assertFalse(json.loads(target.read_text())["simulated"])

    def test_real_snapshot_validates_identity_and_retains_sensor_state(self):
        values = {REG["magic"]:MAGIC, REG["version"]:VERSION, REG["caps"]:7,
                  REG["generation"]:42, REG["behavior"]:1 << 15}
        bridge = Mock(); bridge.rd.side_effect = lambda offset: values.get(offset, 0)
        with tempfile.TemporaryDirectory() as folder:
            value = publish_vehicle(bridge, Path(folder)/"status.json")
            self.assertTrue(value["online"])
            self.assertTrue(value["sensor_stale"])
            self.assertEqual(value["generation"], 42)

if __name__ == "__main__": unittest.main()
