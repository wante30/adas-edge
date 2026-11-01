import os,sys,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"ai"))
from auto_brake import AutoBrakeController
from risk_events import BRAKE_TYPES
from control_scenarios import crosswalk_pulse_candidate
class Journal:
    def __init__(self):self.rows=[]
    def write(self,stage,**row):self.rows.append(dict(stage=stage,**row))
class ControlEligibilityTest(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{"ADAS_CONTROL_MODE":"DRY_RUN"});self.env.start()
        self.net=patch("socket.socket",side_effect=AssertionError("network forbidden"));self.net.start()
        self.journal=Journal();self.c=AutoBrakeController(enabled=True,journal=self.journal)
        self.now=1000000000
        self.event=dict(kind="FRONT_COLLISION_RISK",event_id="test:1",track_id=1,source_id="Q1",severity="critical",valid=True,lifecycle_state="ACTIVE",control_eligible=True,path_valid=True,timestamp_ns=self.now,expires_at_ns=self.now+400000000,inference_mode="ROAD_Q1",mode_epoch="1")
    def tearDown(self):self.c.close();self.net.stop();self.env.stop()
    def update(self,event=None,**kw):return self.c.update(SimpleNamespace(events=[] if event is None else [event]),now_ns=self.now,**kw)
    def test_only_active_eligible_critical_requests(self):
        for updates in [dict(severity="warning"),dict(lifecycle_state="CLEARED"),dict(control_eligible=False),dict(path_valid=False),dict(predicted=True),dict(source_id="Q2"),dict(timestamp_ns=0),dict(valid=False)]:
            self.assertIsNone(self.update(dict(self.event,**updates)))
        self.assertTrue(self.update(self.event)["accepted"]);self.assertFalse(self.c.last_result["executed"])
    def test_all_registered_critical_types_reach_decision(self):
        for i,kind in enumerate(BRAKE_TYPES-{"CROSSWALK_BRAKE_PULSE"}):
            self.assertTrue(self.update(dict(self.event,kind=kind,event_id="event:"+str(i)))["accepted"])
    def test_context_vehicle_and_authorization_rejections(self):
        self.assertIsNone(self.update(self.event,context=dict(mode="MOSAIC_FULL",epoch="1")))
        self.assertIsNone(self.update(self.event,vehicle=dict(available=False)))
        self.c.authorized=False;self.assertIsNone(self.update(self.event))
        self.assertEqual(self.journal.rows[-1]["reason"],"control_not_authorized")
    def test_pulse_request_ack_end_and_no_repeat(self):
        event=crosswalk_pulse_candidate(dict(self.event,kind="PEDESTRIAN_CROSSING_PATH",synthetic=True,evidence="MTL_CROSSWALK_AND_MEASURED_LATERAL_PATH_ENTRY"))
        self.assertTrue(self.update(event)["accepted"])
        self.now+=350000000;self.update(event)
        self.assertEqual(self.c.last_result["stage"],"pulse_ended")
        self.assertEqual(sum(r["stage"]=="brake_requested" for r in self.journal.rows),1)
        self.now+=100000000;self.update()
        stages={r["stage"] for r in self.journal.rows}
        self.assertTrue({"brake_requested","brake_accepted","pulse_simulated_execution","pulse_ended","control_event_cleared"}<=stages)
    def test_nonblocking_timeout(self):
        self.c.sock.recvfrom=lambda size: (_ for _ in ()).throw(BlockingIOError())
        started=time.perf_counter();self.update(self.event)
        self.assertLess(time.perf_counter()-started,.01)
        self.now+=300000000;self.update()
        self.assertEqual(self.c.last_result["stage"],"brake_timeout")
