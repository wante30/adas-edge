"""Model-control unit checks use the real bridge methods without loading RKNN."""
import importlib.util
import os
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import Mock

AI = Path(__file__).parents[1] / "ai"
sys.path.insert(0, str(AI))
os.environ["ADAS_EDGE_ROOT"] = str(AI.resolve())
spec = importlib.util.spec_from_file_location("isolated_model_bridge", AI / "ai_bridge.py")
bridge_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge_module
spec.loader.exec_module(bridge_module)
from model_profiles import build_model_profiles, decode_model_switch_reply


class ModelSwitchTests(unittest.TestCase):
    def make_bridge(self, mode):
        bridge = bridge_module.AiReplayBridge.__new__(bridge_module.AiReplayBridge)
        bridge.model_switch_lock = threading.Lock()
        bridge.active_model_id = "MTL-FAST"
        bridge.model_switch_state = "READY"
        bridge.model_switch_message = ""
        bridge.pending_model_switch = bridge.model_switch_peer = None
        bridge.model_profiles = {name: types.SimpleNamespace(available=True, task=task)
                                 for name, task in [("MTL-FAST", "mtl"), ("FAST-MOSAIC", "mosaic")]}
        bridge.modes = types.SimpleNamespace(mode=mode, adapter=object() if mode == "MOSAIC_FULL" else None)
        bridge.modes.status = lambda: {"mode": bridge.modes.mode, "epoch": "10"}
        bridge.result_socket = Mock()
        return bridge

    def test_mosaic_is_registered_once_with_mosaic_task(self):
        profiles = build_model_profiles("missing", "legacy")
        mosaic = [p for p in profiles if p.id == "FAST-MOSAIC"]
        self.assertEqual(len(mosaic), 1)
        self.assertEqual(mosaic[0].task, "mosaic")
        self.assertTrue({"B", "B-capi", "MTL-FAST", "MTL-FULL", "A", "C", "D"} <= {p.id for p in profiles})

    def test_mvm_ack_reports_actual_mosaic_and_retains_last_road_model(self):
        bridge = self.make_bridge("MOSAIC_FULL")
        bridge._queue_model_switch({"request_id": 12, "profile_id": "FAST-MOSAIC"}, ("local", 1))
        reply = decode_model_switch_reply(bridge.result_socket.sendto.call_args.args[0])
        self.assertEqual(reply["state"], "APPLIED")
        self.assertEqual(reply["active_id"], "FAST-MOSAIC")
        self.assertEqual(bridge.active_model_id, "MTL-FAST")

    def test_pending_mosaic_survives_mode_handshake_and_ack_uses_actual_backend(self):
        bridge = self.make_bridge("ROAD_Q1")
        bridge._queue_model_switch({"request_id": 13, "profile_id": "FAST-MOSAIC"}, ("local", 1))
        self.assertFalse(bridge._apply_pending_model_switch())
        bridge.modes.mode = "MOSAIC_FULL"; bridge.modes.adapter = object()
        bridge._cancel_pending_road_model_switch()
        self.assertTrue(bridge._apply_pending_model_switch())
        reply = decode_model_switch_reply(bridge.result_socket.sendto.call_args.args[0])
        self.assertEqual((reply["state"], reply["active_id"]), ("APPLIED", "FAST-MOSAIC"))

    def test_road_request_waits_for_road_epoch_instead_of_loading_into_mosaic(self):
        bridge = self.make_bridge("MOSAIC_FULL")
        bridge._queue_model_switch({"request_id": 14, "profile_id": "MTL-FAST"}, ("local", 1))
        self.assertEqual(bridge.pending_model_switch["profile_id"], "MTL-FAST")
        self.assertFalse(bridge._apply_pending_model_switch())
