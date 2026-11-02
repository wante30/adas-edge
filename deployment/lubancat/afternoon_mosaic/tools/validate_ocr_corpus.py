"""Capture actual Q2 cards, then compare identical inputs at two OCR padding widths."""
import argparse,json,os,struct,sys,time,zlib
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"ai"))
from q2_navigation_reader import Q2NavigationReader

def main():
 p=argparse.ArgumentParser();p.add_argument("action",choices=["capture","compare"]);p.add_argument("--count",type=int,default=24);args=p.parse_args()
 folder=ROOT/"evidence/ocr_corpus";folder.mkdir(exist_ok=True)
 if args.action=="capture":
  for i in range(args.count):
   data=Path("/dev/shm/pango-video/latest.pgv").read_bytes();header=data[:40];payload=data[40:]
   assert header[:4]==b"PGVF" and zlib.crc32(payload)==struct.unpack_from("<I",header,24)[0]
   packed=np.frombuffer(payload,np.uint8).reshape(720,1280,2)
   image=cv2.resize(cv2.cvtColor(packed[:360,640:],cv2.COLOR_BGR5652BGR),(960,540))
   cv2.imwrite(str(folder/(str(i)+".png")),image)
   time.sleep(2)
 else:
  cv2.setNumThreads(1)
  from rapidocr import RapidOCR
  output={}
  for width in (320,128):
   reader=Q2NavigationReader();reader._ocr=RapidOCR(params={"Global.log_level":"error","Global.use_det":False,"Global.use_cls":False,"Rec.rec_img_shape":[3,48,width],"EngineConfig.onnxruntime.intra_op_num_threads":1,"EngineConfig.onnxruntime.inter_op_num_threads":1});reader.warmup()
   rows=[]
   for i,path in enumerate(sorted(folder.glob("*.png"),key=lambda p:int(p.stem))):
    image=cv2.imread(str(path));start=time.monotonic_ns();cpu=time.process_time_ns();r=reader.update(image,i+1,start)
    rows.append(dict(file=path.name,wall_ms=(time.monotonic_ns()-start)/1e6,cpu_ms=(time.process_time_ns()-cpu)/1e6,result=r.to_dict()))
   output[str(width)]=rows
  output["differences"]=[dict(file=a["file"],before=a["result"],after=b["result"]) for a,b in zip(output["320"],output["128"]) if any(a["result"].get(k)!=b["result"].get(k) for k in ("valid","distance_m","direction"))]
  (ROOT/"evidence/ocr_corpus_comparison.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
  print("OCR_CORPUS",len(output["320"]),"differences",len(output["differences"]),flush=True)
if __name__=="__main__":main()
