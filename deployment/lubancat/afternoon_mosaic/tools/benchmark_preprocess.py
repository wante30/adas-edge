import json,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"ai"))
from q2_navigation_reader import Q2NavigationReader
from tools.profile_rknn_single_stream import letterbox_bgr
image=cv2.imread(str(ROOT/"evidence/ocr_input.png"));road=cv2.resize(image,(640,360));out=[]
for threads in (4,1):
 cv2.setNumThreads(threads)
 for name,fn in (("letterbox",lambda:letterbox_bgr(road,(512,288))),
                 ("card_locate",lambda:Q2NavigationReader._locate_card(image))):
  fn();start=time.monotonic_ns();cpu=time.process_time_ns()
  for _ in range(60):fn()
  out.append(dict(threads=threads,stage=name,wall_ms=(time.monotonic_ns()-start)/60e6,cpu_ms=(time.process_time_ns()-cpu)/60e6))
(ROOT/"evidence/preprocess_threads.json").write_text(json.dumps(out,indent=2));print(out)
