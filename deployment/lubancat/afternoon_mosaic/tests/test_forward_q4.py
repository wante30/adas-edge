import os,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"ai"))
from primary_tracks import PrimaryTracks
from light_risk import LightRiskEngine
from inference_mode import MosaicRoadAdapter
from risk_events import EventStore
from auto_brake import AutoBrakeController

class Q4ForwardTest(unittest.TestCase):
    def test_road_track_identity_is_q4(self):
        track=PrimaryTracks().update([dict(bbox=[260,170,380,320],class_name="car",confidence=.9)],
                                    ("ROAD_Q1",1,"MTL",3),1000000000)[0]
        self.assertEqual(track["source_id"],"Q4")
        self.assertGreater(track["track_id"],0)

    def test_mosaic_adapter_forwards_only_q4_in_local_coordinates(self):
        q1=dict(bbox=[260,170,380,320],class_name="car",confidence=.9)
        q4=dict(q1,bbox=[900,530,1020,680])
        detector=SimpleNamespace(detect=lambda frame:([q1,q4,dict(q1,bbox=[600,300,700,400])],12))
        adapter=MosaicRoadAdapter(detector);adapter.road_quadrant=3;adapter.full_frame=object()
        selected,_=adapter.detect(None)
        self.assertEqual(len(selected),1);self.assertEqual(selected[0]["bbox"],q1["bbox"])
        self.assertEqual(adapter.all_detections[1]["bbox"],q4["bbox"])

    def test_q4_risk_mapping_and_control_in_both_modes(self):
        for mode in ("ROAD_Q1","MOSAIC_FULL"):
            engine=LightRiskEngine();store=EventStore()
            candidates=[]
            for i in range(10):
                now=1000000000+i*100000000;h=.18+i*.035
                box=[278,(.9-h)*360,362,324]
                if mode=="MOSAIC_FULL":box=[box[0]+640,box[1]+360,box[2]+640,box[3]+360]
                d=dict(bbox=box,class_name="car",confidence=.9,source_id="Q4",track_id=7)
                candidates=engine.update([d],now,(mode,1,"MODEL",3))
            self.assertTrue(any(c["severity"]=="critical" for c in candidates))
            events=store.update(candidates,mode=mode,epoch=1,bridge_id="b",model_id="MODEL",frame_id=9,
                timestamp_ns=now,now_ns=now,road_quadrant=3)
            e=next(e for e in events if e["severity"]=="critical")
            self.assertEqual(e["source_id"],"Q4");self.assertTrue(e["control_eligible"])
            self.assertEqual(e["bbox"][0],556 if mode=="ROAD_Q1" else 918)
            with patch.dict(os.environ,{"ADAS_CONTROL_MODE":"DRY_RUN"}):
                ctrl=AutoBrakeController(enabled=True,forward_source_id="Q4")
                self.assertIsNone(ctrl._reason(e,now,None,dict(available=True,authorized=True)))
                self.assertEqual(ctrl._reason(dict(e,source_id="Q1"),now,None,dict(available=True,authorized=True)),"not_measured_forward_target")
                ctrl.close()

    def test_non_forward_quadrants_do_not_generate_q4_braking_candidates(self):
        engine=LightRiskEngine()
        for source in ("Q1","Q2","Q3"):
            self.assertEqual(engine.update([dict(source_id=source,track_id=1)],1,("MOSAIC_FULL",1,"FAST",3)),[])

if __name__=="__main__":unittest.main()
