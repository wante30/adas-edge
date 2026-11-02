"""Compare CRC-validated cached DMA snapshots with the SAME published frame."""
import argparse, hashlib, json, re, struct, zlib
from pathlib import Path
import numpy as np

def quadrants(payload, previous=None):
    pixels=np.frombuffer(payload,dtype="<u2").reshape(720,1280)
    old=np.frombuffer(previous,dtype="<u2").reshape(720,1280) if previous else None
    result=[]
    for q in range(4):
        y,x=(q//2)*360,(q%2)*640
        a=pixels[y:y+360,x:x+640]
        result.append(dict(source_id="Q%d"%(q+1),nonzero_pixels=int(np.count_nonzero(a)),
            nonzero_fraction=float(np.count_nonzero(a)/a.size),
            changed_pixels=int(np.count_nonzero(a!=old[y:y+360,x:x+640])) if old is not None else None,
            crc32="%08x"%(zlib.crc32(a.tobytes())&0xffffffff)))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("directory",type=Path);a=p.parse_args();d=a.directory
    log=(d/"capture.log").read_text()
    frames=[];previous=None
    for frame_id,generation,crc in re.findall(r"frame=(\d+) generation=(\d+).*crc32=([0-9a-f]+)",log):
        raw=(d/"raw"/("frame_g%s_%010d.rgb565"%(generation,int(frame_id)))).read_bytes()
        actual=zlib.crc32(raw)&0xffffffff
        frames.append(dict(frame_id=int(frame_id),generation=int(generation),crc_valid=actual==int(crc,16),
                           quadrants=quadrants(raw,previous)))
        previous=raw
    published=(d/"published/latest.pgv").read_bytes()
    magic,version,w,h,fid,gen,crc,length,lo,hi=struct.unpack("<4s9I",published[:40])
    raw=(d/"raw"/("frame_g%d_%010d.rgb565"%(gen,fid))).read_bytes()
    report=dict(kind="SAME_FRAME_VALIDATED_DMA_TO_PGVF",frames=frames,
        published=dict(frame_id=fid,generation=gen,crc32="%08x"%crc,
          header_valid=(magic==b"PGVF" and version==1 and (w,h)==(1280,720) and length==len(raw)),
          crc_valid=(zlib.crc32(published[40:])&0xffffffff)==crc,
          payload_equals_cached_dma=(published[40:]==raw),quadrants=quadrants(published[40:]),
          sha256=hashlib.sha256(published).hexdigest()),
        conclusion="Inspect validated DMA quadrant data; this test does not distinguish camera input from FPGA compositor routing.")
    report["passed_integrity"]=bool(frames and all(f["crc_valid"] for f in frames) and
        all(report["published"][k] for k in ("header_valid","crc_valid","payload_equals_cached_dma")))
    (d/"quadrant_diagnosis.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(dict(frames=len(frames),published=report["published"],passed_integrity=report["passed_integrity"])))
    return 0 if report["passed_integrity"] else 1
if __name__=="__main__":raise SystemExit(main())
