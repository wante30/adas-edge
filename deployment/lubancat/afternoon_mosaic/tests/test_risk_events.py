import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/"ai"))
from risk_events import EventStore

class Journal:
    def __init__(self): self.rows=[]
    def write(self, stage, **row): self.rows.append(dict(stage=stage,**row))

class RiskEventTests(unittest.TestCase):
    def setUp(self):
        self.journal=Journal(); self.store=EventStore(self.journal); self.frame=0
    def step(self, candidates=(), *, epoch=1, source="Q1", age=0):
        self.frame+=1; now=1_000_000_000+self.frame*100_000_000
        rows=[dict(c,source_id=source) for c in candidates]
        return self.store.update(rows,mode="ROAD_Q1",epoch=epoch,bridge_id="bridge-id",model_id="MTL-FAST",frame_id=self.frame,timestamp_ns=now,measurement_ns=now-age,now_ns=now)
    def candidate(self, severity):
        return {"kind":"FRONT_COLLISION_RISK","severity":severity,"track_id":7,"target_class":"car","bbox":[260,170,380,320]}
    def test_warning_never_control_eligible_and_road_box_is_scaled_once(self):
        event=self.step([self.candidate("warning")])[0]
        self.assertFalse(event["control_eligible"]); self.assertEqual(event["bbox"],[520,340,760,640])
    def test_identity_survives_upgrade_and_downgrade(self):
        a=self.step([self.candidate("warning")])[0];b=self.step([self.candidate("critical")])[0]
        self.assertEqual(a["event_id"],b["event_id"]);self.assertTrue(b["control_eligible"])
        c=self.step([self.candidate("warning")])[0];self.assertFalse(c["control_eligible"])
    def test_clear_after_two_misses_without_ui_latch(self):
        event=self.step([self.candidate("critical")])[0]
        self.assertEqual(len(self.step()),1);self.assertEqual(self.step(),())
        cleared=self.store.updates[-1];self.assertEqual(cleared["event_id"],event["event_id"])
        self.assertEqual(cleared["lifecycle_state"],"CLEARED");self.assertFalse(cleared["control_eligible"])
    def test_epoch_change_clears_old_identity(self):
        a=self.step([self.candidate("critical")])[0];b=self.step([self.candidate("critical")],epoch=2)[0]
        self.assertNotEqual(a["event_id"],b["event_id"]);self.assertEqual(self.store.updates[0]["lifecycle_state"],"CLEARED")
    def test_stale_measurement_is_not_refreshed_by_new_detection_frame(self):
        self.assertEqual(self.step([self.candidate("critical")],age=500_000_000),())
    def test_other_quadrant_cannot_enter_control(self):
        self.assertFalse(self.step([self.candidate("critical")],source="Q2")[0]["control_eligible"])
