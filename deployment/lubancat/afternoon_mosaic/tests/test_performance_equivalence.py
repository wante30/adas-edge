import sys,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/"ai"))
import capi_detector as capi
import tools.competition_console.byte_tracker_lite as tracker

class PerformanceEquivalence(unittest.TestCase):
    def test_overlay_palette_is_byte_identical_for_all_semantic_classes(self):
        lane,style,drive=np.indices((9,3,3),dtype=np.uint8)
        canvas=np.zeros((9,3,3,4),np.uint8)
        canvas[drive==capi.DRIVE_DIRECT]=capi.COLOR_DIRECT
        canvas[drive==capi.DRIVE_ALTERNATIVE]=capi.COLOR_ALTERNATIVE
        fg=lane!=capi.LANE_BACKGROUND
        canvas[fg & (style==capi.STYLE_DASHED)]=capi.COLOR_DASHED
        canvas[fg & (style!=capi.STYLE_DASHED) & (lane!=capi.LANE_CROSSWALK)]=capi.COLOR_SOLID
        canvas[lane==capi.LANE_CROSSWALK]=capi.COLOR_CROSSWALK
        np.testing.assert_array_equal(capi.overlay_colors(lane,style,drive),canvas)

    def test_greedy_assignment_does_not_match_wrong_class_or_reuse_a_track(self):
        tracks=[SimpleNamespace(bbox=(0,0,100,100),family="vehicle"),
                SimpleNamespace(bbox=(200,0,230,100),family="person")]
        detections=[dict(x=2,y=0,width=100,height=100,className="car"),
                    dict(x=4,y=0,width=100,height=100,className="car"),
                    dict(x=200,y=0,width=30,height=100,className="car")]
        with patch.object(tracker,"linear_sum_assignment",None):
            matched,unmatched_t,unmatched_d=tracker.ByteTracker()._associate(tracks,detections,.8)
        self.assertEqual(matched,[(tracks[0],detections[0])])
        self.assertEqual(unmatched_t,[1])
        self.assertEqual(unmatched_d,[1,2])

if __name__=="__main__":unittest.main()
