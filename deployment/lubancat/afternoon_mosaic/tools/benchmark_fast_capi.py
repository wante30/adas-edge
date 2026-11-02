"""Isolated C API upper-bound on one CRC-verified real PCIe snapshot, no Qt/OCR."""
import json,os,struct,sys,time,zlib
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"ai"))
from fast_mosaic_backend import FastMosaicDetector
cv2.setNumThreads(1)
data=Path("/dev/shm/pango-video/latest.pgv").read_bytes();header=data[:40];payload=data[40:]
assert header[:4]==b"PGVF" and zlib.crc32(payload)==struct.unpack_from("<I",header,24)[0]
packed=np.frombuffer(payload,np.uint8).reshape(720,1280,2);detector=FastMosaicDetector();out={}
for mode in ("FAST-Q4","FAST-MOSAIC"):
 rows=[]
 for i in range(55):
  start=time.monotonic_ns();cpu=time.process_time_ns();input_frame=cv2.cvtColor(packed[360:,640:] if mode=="FAST-Q4" else packed,cv2.COLOR_BGR5652BGR);crop=(time.monotonic_ns()-start)/1e6
  detected,elapsed=detector.detect(input_frame)
  if i>=5:rows.append(dict(wall_ms=(time.monotonic_ns()-start)/1e6,cpu_ms=(time.process_time_ns()-cpu)/1e6,crop_ms=crop,timing=detector.last_timing,boxes=len(detected)))
 out[mode]=dict(fps=len(rows)*1000/sum(r["wall_ms"] for r in rows),rows=rows)
detector.close()
out["source"]=dict(kind="REAL_PCIE_FIXED_SNAPSHOT_NO_QT_NO_OCR",frame_id=struct.unpack_from("<I",header,16)[0],timestamp_ns=struct.unpack_from("<Q",header,32)[0],crc32=struct.unpack_from("<I",header,24)[0])
(ROOT/"evidence/standalone_capi.json").write_text(json.dumps(out,indent=2));print({k:v["fps"] for k,v in out.items() if k.startswith("FAST")})
