"""FAST-Q4 uses packed physical Q4, the shared ABI and fresh semantics."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

AI = Path(__file__).parents[1]/"ai"
sys.path.insert(0,str(AI))
os.environ["ADAS_EDGE_ROOT"]=str(AI.resolve())
spec=importlib.util.spec_from_file_location("fast_q4_bridge_test",AI/"ai_bridge.py")
bridge=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=bridge
spec.loader.exec_module(bridge)
from model_profiles import build_model_profiles
from light_semantics import SemanticWorker
from fast_mosaic_backend import FastMosaicDetector

class FastQ4Test(unittest.TestCase):
    def test_packed_crop_preserves_stride_colors_and_does_not_decode_full_frame(self):
        packed=np.zeros((720,1280),dtype="<u2")
        packed[:360,:640]=0xf800
        packed[:360,640:]=0x07e0
        packed[360:,:640]=0x001f
        packed[360:,640:]=0xffff
        frame=bridge.PackedRgb565(packed.tobytes())
        with patch.object(bridge.PackedRgb565,"full_bgr",side_effect=AssertionError("full frame decode")):
            q4,q2=bridge.select_inference_frames(frame,"quad",3,1)
        self.assertEqual(q4.shape,(360,640,3))
        self.assertTrue(np.all(q4==[248,252,248]))
        self.assertTrue(np.all(q2==[0,252,0]))

    def test_profile_is_single_detection_and_uses_independent_generic_backend(self):
        profiles={p.id:p for p in build_model_profiles("missing","legacy")}
        self.assertEqual((profiles["FAST-Q4"].task,profiles["FAST-Q4"].backend),("detect","fast_mosaic"))
        self.assertEqual(profiles["FAST-MOSAIC"].task,"mosaic")
        with patch("fast_mosaic_backend.available",return_value=False):
            self.assertFalse(profiles["FAST-Q4"].available)

    def test_shared_detector_rescales_capi_results_from_single_input(self):
        detector=FastMosaicDetector.__new__(FastMosaicDetector)
        detector.backend="FAST-Q4";detector.output_count=9;detector.scale_count=3
        class Inner:
            last_timing={"total_ms":40}
            def detect(self,image):
                self.shape=image.shape
                return [{"bbox":[260,170,380,320],"class_name":"car"}],40
        detector.inner=Inner()
        rows,_=detector.detect(np.zeros((360,640,3),np.uint8))
        self.assertEqual(detector.inner.shape,(360,640,3))
        self.assertEqual(rows[0]["bbox"],[260,170,380,320])
        self.assertEqual(detector.last_timing["backend"],"FAST-Q4")

    def test_previous_mtl_semantics_are_unavailable_after_fast_q4_switch(self):
        worker=SemanticWorker()
        try:
            old=("ROAD_Q1",1,"MTL-FAST",3);new=("ROAD_Q1",1,"FAST-Q4",3)
            worker.result={"context":old,"timestamp_ns":100,"mtl_available":True,"crosswalks":[{}]}
            self.assertEqual(worker.current(101,new),{})
            result=worker._process(np.zeros((360,640,3),np.uint8),[],102,new,None,None)
            self.assertFalse(result["mtl_available"])
            self.assertEqual(result["crosswalks"],[])
        finally:worker.close()

if __name__=="__main__":unittest.main()
