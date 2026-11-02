"""Compare OCR thread counts on one CRC-verified actual PCIe Q2 image."""
import argparse,json,os,struct,sys,time,zlib
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"ai"))
from q2_navigation_reader import Q2NavigationReader

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--threads",type=int,required=True);p.add_argument("--width",type=int,default=320);p.add_argument("--legacy-affinity",action="store_true");p.add_argument("--image",required=True);args=p.parse_args()
 imagepath=Path(args.image)
 if not imagepath.exists():
  data=Path("/dev/shm/pango-video/latest.pgv").read_bytes();header=data[:40];payload=data[40:]
  assert header[:4]==b"PGVF" and zlib.crc32(payload)==struct.unpack_from("<I",header,24)[0]
  packed=np.frombuffer(payload,np.uint8).reshape(720,1280,2)
  image=cv2.resize(cv2.cvtColor(packed[:360,640:],cv2.COLOR_BGR5652BGR),(960,540))
  cv2.imwrite(str(imagepath),image)
 image=cv2.imread(str(imagepath));reader=Q2NavigationReader()
 from rapidocr import RapidOCR
 if args.legacy_affinity:os.sched_setaffinity(0,{(os.cpu_count() or 4)-1})
 cv2.setNumThreads(1)
 reader._ocr=RapidOCR(params={"Global.log_level":"error","Global.use_det":False,"Global.use_cls":False,"Rec.rec_img_shape":[3,48,args.width],
  "EngineConfig.onnxruntime.intra_op_num_threads":args.threads,"EngineConfig.onnxruntime.inter_op_num_threads":1})
 reader.warmup();rows=[]
 for i in range(8):
  start=time.monotonic_ns();cpu=time.process_time_ns();result=reader.update(image,i+1,start)
  rows.append(dict(wall_ms=(time.monotonic_ns()-start)/1e6,cpu_ms=(time.process_time_ns()-cpu)/1e6,result=result.to_dict()))
 print(json.dumps(dict(threads=args.threads,width=args.width,affinity=sorted(os.sched_getaffinity(0)),rows=rows)),flush=True)
