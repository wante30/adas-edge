import importlib.util
from pathlib import Path
import struct
import types
import unittest

spec = importlib.util.spec_from_file_location("inference_mode", Path(__file__).parents[1] / "ai/inference_mode.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Socket:
    def __init__(self): self.messages = []
    def sendto(self, payload, peer): self.messages.append((payload, peer))

class Detector:
    def __init__(self): self.calls = []
    def detect(self, frame):
        self.calls.append(frame)
        return [{"class_name": "car", "bbox": b} for b in
                ([20,30,200,300], [700,50,900,300], [50,400,300,600],
                 [700,400,1000,600], [620,200,680,400])], 10

class ModeTests(unittest.TestCase):
    def setUp(self):
        self.detector = Detector()
        self.resets = 0
        def reset(): self.resets += 1
        self.bridge = types.SimpleNamespace(result_socket=Socket(),
            _cancel_pending_road_model_switch=lambda: None,
            pipeline=types.SimpleNamespace(main_detector=self.detector, specialist_detector=None, reset=reset))
        self.control = m.ModeControl(self.bridge, mosaic_factory=lambda road: road)
    def command(self, epoch, mode):
        self.control.receive(m.COMMAND.pack(b"MVI1", epoch, mode), ("127.0.0.1",1234))
    def test_idempotence_and_older_epoch(self):
        self.command(10,1); self.control.apply()
        adapter=self.control.adapter
        self.command(10,1); self.control.apply()
        self.assertIs(adapter,self.control.adapter)
        self.assertEqual(self.resets,1)
        self.command(9,0); self.control.apply()
        self.assertEqual(self.control.mode,m.MOSAIC)
        self.command(11,0); self.control.apply()
        self.assertIs(self.bridge.pipeline.main_detector,self.detector)
    def test_old_quadrant_packet_cannot_change_mode(self):
        self.command(10,1); self.control.apply()
        self.assertFalse(self.control.receive(b"MVS1\x00\x01",("127.0.0.1",1234)))
        self.assertEqual(self.control.mode,m.MOSAIC)
    def test_request_epoch_is_required(self):
        self.command(10,0); self.control.apply()
        request=lambda epoch:m.REQUEST.pack(b"MVR3",8,100,epoch,60.,30.)
        self.assertTrue(self.control.frame_request(request(10)))
        self.assertFalse(self.control.frame_request(request(9)))
        self.command(11,1)
        self.assertFalse(self.control.frame_request(request(10)))
    def test_control_packets_pass_through_after_handshake(self):
        self.command(10,0); self.control.apply()
        for packet in (b"MVM1model", b"MVS1\x00\x01"):
            self.assertEqual(self.control.frame_request(packet), packet)
        self.assertEqual(self.control.frame_request(b"MVR2oldframe"), b"")
    def test_owned_full_frame_context_released_once(self):
        detector=Detector()
        detector.closed=0
        def close(): detector.closed+=1
        detector.close=close
        adapter=m.MosaicRoadAdapter(detector, owns_detector=True)
        adapter.close(); adapter.close()
        self.assertEqual(detector.closed,1)
    def test_mosaic_load_failure_retains_road_detector(self):
        def fail(_): raise RuntimeError("bad library")
        self.control.mosaic_factory=fail
        self.command(10,1); self.control.apply()
        self.assertEqual(self.control.mode,m.ROAD)
        self.assertIs(self.bridge.pipeline.main_detector,self.detector)
        self.assertIn(b'"state":"FAILED"',self.bridge.result_socket.messages[-1][0])
    def test_full_frame_once_and_only_q1_enters_road_logic(self):
        adapter=m.MosaicRoadAdapter(self.detector)
        full=object(); adapter.full_frame=full
        rows,_=adapter.detect(object())
        self.assertEqual(self.detector.calls,[full])
        self.assertEqual(len(adapter.all_detections),5)
        self.assertEqual([r["bbox"] for r in rows],[[20,30,200,300]])

if __name__ == "__main__": unittest.main()
