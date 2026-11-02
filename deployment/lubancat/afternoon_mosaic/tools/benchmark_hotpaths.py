import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"ai"))
import capi_detector as c
rng=np.random.default_rng(4);lane=rng.integers(0,9,(144,256),dtype=np.uint8);style=rng.integers(0,3,(144,256),dtype=np.uint8);drive=rng.integers(0,3,(144,256),dtype=np.uint8)
def old_overlay():
 a=np.zeros((*lane.shape,4),np.uint8);a[drive==c.DRIVE_DIRECT]=c.COLOR_DIRECT;a[drive==c.DRIVE_ALTERNATIVE]=c.COLOR_ALTERNATIVE
 fg=lane!=c.LANE_BACKGROUND;a[fg & (style==c.STYLE_DASHED)]=c.COLOR_DASHED;a[fg & (style!=c.STYLE_DASHED) & (lane!=c.LANE_CROSSWALK)]=c.COLOR_SOLID;a[lane==c.LANE_CROSSWALK]=c.COLOR_CROSSWALK
 return a
cost=rng.uniform(0,2,(45,12)).astype(np.float32)
def new_candidates():
 order=np.argsort(cost,axis=None);return order[cost.ravel()[order]<=.8].tolist()
old_candidates=lambda:[int(k) for k in np.argsort(cost,axis=None) if cost.flat[k]<=.8]
assert old_candidates()==new_candidates();np.testing.assert_array_equal(old_overlay(),c.overlay_colors(lane,style,drive))
result={}
for name,fn in (("overlay_old",old_overlay),("overlay_lut",lambda:c.overlay_colors(lane,style,drive)),("track_old",old_candidates),("track_vector_filter",new_candidates)):
 start=time.monotonic_ns();cpu=time.process_time_ns()
 for i in range(100):fn()
 result[name]=dict(wall_ms=(time.monotonic_ns()-start)/100e6,cpu_ms=(time.process_time_ns()-cpu)/100e6)
(ROOT/"evidence/hotpath_benchmark.json").write_text(json.dumps(result,indent=2));print(result)
