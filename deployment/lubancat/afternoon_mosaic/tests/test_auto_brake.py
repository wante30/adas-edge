import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"ai"))
from auto_brake import AutoBrakeController

class ControlEligibilityTest(unittest.TestCase):
    def test_only_active_eligible_critical_requests(self):
        with patch.dict(os.environ,{"ADAS_CONTROL_MODE":"DRY_RUN"}), patch("socket.socket",side_effect=AssertionError("network forbidden")):
            control=AutoBrakeController(enabled=True)
            event=dict(kind="FRONT_COLLISION_RISK",event_id="test:1",track_id=1,severity="critical",lifecycle_state="ACTIVE",control_eligible=True,path_valid=True)
            for updates in [dict(severity="warning"),dict(lifecycle_state="CLEARED"),dict(control_eligible=False),dict(path_valid=False)]:
                self.assertIsNone(control.update(SimpleNamespace(events=[dict(event,**updates)])))
            self.assertTrue(control.update(SimpleNamespace(events=[event]))["accepted"])
            self.assertFalse(control.last_result["executed"])
