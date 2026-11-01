import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"ai"))
from primary_tracks import PrimaryTracks
class IdentityTest(unittest.TestCase):
    def test_source_epoch_and_stability(self):
        tracker=PrimaryTracks();ctx=("MOSAIC_FULL","1","FAST-MOSAIC")
        rows=[dict(bbox=[100,100,200,200],class_name="car",confidence=.9),dict(bbox=[740,100,840,200],class_name="car",confidence=.9)]
        a=tracker.update(rows,ctx,1000000000)
        b=tracker.update(rows,ctx,1100000000)
        self.assertEqual([r["track_id"] for r in a],[r["track_id"] for r in b])
        self.assertNotEqual(a[0]["track_id"],a[1]["track_id"])
        self.assertEqual([r["source_id"] for r in a],["Q1","Q2"])
        c=tracker.update(rows,("MOSAIC_FULL","2","FAST-MOSAIC"),1200000000)
        self.assertNotEqual(a[0]["track_id"],c[0]["track_id"])
    def test_seam_cannot_control(self):
        rows=PrimaryTracks().update([dict(bbox=[600,100,680,200],class_name="car",confidence=.9)],("MOSAIC_FULL","1","FAST"),1)
        self.assertEqual(rows[0]["track_id"],0)
        self.assertEqual(rows[0]["source_id"],"UNASSIGNED")

    def test_time_expiry_and_missing_measurements(self):
        tracker=PrimaryTracks();ctx=("ROAD_Q1","1","MTL")
        row=dict(bbox=[100,100,200,200],class_name="car",confidence=.9)
        original=tracker.update([row],ctx,1000000000)[0]["track_id"]
        self.assertEqual(tracker.update([],ctx,1100000000),[])
        self.assertEqual(tracker.update([dict(row,confidence=.4)],ctx,1200000000)[0]["track_id"],original)
        self.assertNotEqual(tracker.update([row],ctx,2000000000)[0]["track_id"],original)
    def test_velocity_is_time_based_and_raw_measurement_preserved(self):
        tracker=PrimaryTracks();ctx=("ROAD_Q1","1","MTL")
        for i in range(12):
            bbox=[100+i*3,100,200+i*3,200]
            rows=tracker.update([dict(bbox=bbox,class_name="car",confidence=.9)],ctx,1000000000+i*100000000)
            self.assertEqual(rows[0]["bbox"],bbox)
        self.assertGreater(rows[0]["velocity"][0],0)
        self.assertNotIn("predicted",rows[0])
