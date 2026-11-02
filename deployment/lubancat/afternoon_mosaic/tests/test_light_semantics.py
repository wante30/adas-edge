import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"ai"))
from light_semantics import SemanticWorker
class SemanticsTest(unittest.TestCase):
    def test_mtl_mask_and_missing_mosaic_semantics(self):
        worker=SemanticWorker()
        try:
            masks=np.full((3,144,256),8,np.uint8);masks[0,90:110,60:190]=0
            frame=np.zeros((360,640,3),np.uint8)
            road=worker._process(frame,[],100,("ROAD_Q1",1,"MTL"),masks,(.8,0,0))
            self.assertTrue(road["crosswalks"])
            mosaic=worker._process(frame,[],200,("MOSAIC_FULL",2,"FAST"),None,None)
            self.assertFalse(mosaic["crosswalks"]);self.assertFalse(mosaic["mtl_available"])
            worker.result=road
            self.assertEqual(worker.current(1000000100,("ROAD_Q1",1,"MTL")),{})
            self.assertEqual(worker.current(101,("ROAD_Q1",2,"MTL")),{})
        finally:worker.close()
    def test_semantics_alone_cannot_create_critical(self):
        from light_risk import LightRiskEngine
        engine=LightRiskEngine()
        semantics=dict(crosswalks=[dict(bbox=[.2,.6,.8,.9],confidence=.9)])
        self.assertEqual(engine.update([],100,("ROAD_Q1",1,"MTL"),semantics),[])
    def test_crosswalk_person_association_requires_measured_person(self):
        from light_risk import LightRiskEngine
        engine=LightRiskEngine();events=[]
        d=dict(bbox=[280,200,350,310],source_id="Q1",track_id=1,class_name="person",confidence=.9)
        for i in range(3):events=engine.update([d],1000000000+i*100000000,("ROAD_Q1",1,"MTL"),dict(crosswalks=[dict(bbox=[.2,.6,.8,.9])]))
        self.assertTrue(any(e["kind"]=="PEDESTRIAN_NEAR_CROSSWALK" for e in events))
        self.assertFalse(any(e["kind"]=="PEDESTRIAN_NEAR_CROSSWALK" and e["severity"]=="critical" for e in events))
