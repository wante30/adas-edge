import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/"ai"))
from navigation_gate import NavigationGate

class NavigationGateTest(unittest.TestCase):
    def test_unchanged_card_uses_actual_freshness_without_refreshing_timestamp(self):
        gate=NavigationGate();image=np.zeros((360,640,3),np.uint8)
        self.assertTrue(gate.wants(image,100,True,90))
        gate.submitted(image,100)
        self.assertFalse(gate.wants(image,500_000_000,True,90))
        self.assertTrue(gate.wants(image,1_100_000_000,True,90))
        self.assertEqual(gate.submitted_ns,100)

    def test_map_motion_does_not_force_card_recognition_but_changed_digit_does(self):
        gate=NavigationGate();image=np.zeros((360,640,3),np.uint8)
        gate.submitted(image,100)
        moved=image.copy();moved[230:350,350:600]=255
        self.assertFalse(gate.wants(moved,500,True,90))
        digit=image.copy();digit[125:132,96:105]=255
        self.assertTrue(gate.wants(digit,500,True,90))

    def test_invalid_old_and_future_ocr_are_never_reused(self):
        gate=NavigationGate();image=np.zeros((360,640,3),np.uint8);gate.submitted(image,100)
        self.assertTrue(gate.wants(image,200,False,100))
        self.assertTrue(gate.wants(image,200,True,300))

if __name__=="__main__":unittest.main()
