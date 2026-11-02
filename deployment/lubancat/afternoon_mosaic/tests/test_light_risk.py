import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"ai"))
from light_risk import LightRiskEngine

class FastRulesTest(unittest.TestCase):
    def run_sequence(self,cls="car",heights=None,centers=None,**extra):
        engine=LightRiskEngine();out=[]
        heights=heights or [.35]*10;centers=centers or [.5]*len(heights)
        for i,(h,x) in enumerate(zip(heights,centers)):
            w=.13;foot=.9
            d=dict(bbox=[(x-w/2)*640,(foot-h)*360,(x+w/2)*640,foot*360],class_name=cls,confidence=.9,source_id="Q1",track_id=1,synthetic=True,**extra)
            out+=engine.update([d],1000000000+i*100000000,("ROAD_Q1","1","MTL"))
        return out,engine
    def test_static_large_is_only_warning(self):
        events,_=self.run_sequence()
        self.assertTrue(events);self.assertFalse(any(e["severity"]=="critical" for e in events))
    def test_sustained_growth_confirms_forward_risk(self):
        events,engine=self.run_sequence(heights=[.18+i*.035 for i in range(10)])
        self.assertTrue(any(e["kind"]=="FRONT_COLLISION_RISK" and e["severity"]=="critical" for e in events))
        self.assertGreater(engine.metrics()["front"]["count"],0)
    def test_person_and_two_wheeler(self):
        for cls,kind in [("person","PEDESTRIAN_IN_PATH"),("bicycle","TWO_WHEELER_PATH_CONFLICT"),("motorcycle","TWO_WHEELER_PATH_CONFLICT")]:
            events,_=self.run_sequence(cls=cls)
            self.assertTrue(any(e["kind"]==kind and e["severity"]=="critical" for e in events),cls)
    def test_lateral_crossing_and_cut_in(self):
        for cls,kind in [("car","VEHICLE_CUT_IN"),("person","PEDESTRIAN_CROSSING_PATH")]:
            events,_=self.run_sequence(cls=cls,centers=[.85,.82,.79,.76,.73,.70,.67,.64,.61,.58])
            self.assertTrue(any(e["kind"]==kind for e in events),kind)
    def test_growth_does_not_prove_oncoming(self):
        events,_=self.run_sequence(heights=[.18+i*.035 for i in range(10)],direction="oncoming",direction_verified=False)
        self.assertFalse(any(e["kind"]=="ONCOMING_PATH_CONFLICT" and e["severity"]=="critical" for e in events))
        events,_=self.run_sequence(heights=[.18+i*.035 for i in range(10)],direction="oncoming",direction_verified=True)
        self.assertTrue(any(e["kind"]=="ONCOMING_PATH_CONFLICT" and e["severity"]=="critical" for e in events))
    def test_q2_ignored(self):
        e=LightRiskEngine()
        self.assertEqual(e.update([dict(source_id="Q2",track_id=1)],1,("MOSAIC_FULL",1,"FAST")),[])
