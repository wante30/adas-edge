from __future__ import annotations
import struct
from dataclasses import dataclass
from typing import ClassVar

FRAME_MAGIC=b"ADSF"; RESULT_MAGIC=b"ADSR"; VERSION=1
FRAME_HEADER=struct.Struct("<4sHHIQQII BBH III")
RESULT_HEADER=struct.Struct("<4sHHIQQHBBBBII")
DETECTION=struct.Struct("<HHHHHH")

class ProtocolError(ValueError): pass
def _check_version(v: int) -> None:
    if v != VERSION: raise ProtocolError(f"unsupported version: {v}")

@dataclass(frozen=True)
class FrameDescriptor:
    stream_id:int; frame_id:int; timestamp_ns:int; width:int; height:int; pixel_format:int; stride:int; payload:bytes; flags:int=0
    magic: ClassVar[bytes]=FRAME_MAGIC
    def serialize(self)->bytes:
        header=FRAME_HEADER.pack(self.magic,VERSION,FRAME_HEADER.size,self.stream_id,self.frame_id,self.timestamp_ns,self.width,self.height,self.pixel_format,0,self.stride,len(self.payload),self.flags,0)
        return header+self.payload
    @classmethod
    def deserialize(cls,data:bytes):
        if len(data)<FRAME_HEADER.size: raise ProtocolError("truncated frame header")
        m,v,hlen,sid,fid,ts,w,h,pf,_r,stride,size,flags,_crc=FRAME_HEADER.unpack_from(data)
        if m!=FRAME_MAGIC: raise ProtocolError("bad frame magic")
        _check_version(v)
        if hlen!=FRAME_HEADER.size or size != len(data)-hlen: raise ProtocolError("invalid frame payload size")
        return cls(sid,fid,ts,w,h,pf,stride,data[hlen:],flags)

@dataclass(frozen=True)
class DetectionRecord:
    class_id:int; confidence:float; x1:int; y1:int; x2:int; y2:int
    def pack(self)->bytes: return DETECTION.pack(self.class_id,max(0,min(65535,round(self.confidence*32767))),self.x1,self.y1,self.x2,self.y2)
    @classmethod
    def unpack(cls,data:bytes):
        cid,conf,x1,y1,x2,y2=DETECTION.unpack(data); return cls(cid,conf/32767.0,x1,y1,x2,y2)

@dataclass(frozen=True)
class ResultDescriptor:
    stream_id:int; frame_id:int; timestamp_ns:int; detections:tuple[DetectionRecord,...]=(); traffic_light_state:int=0; lane_state:int=0; adas_action:int=0; warning_flags:int=0
    magic: ClassVar[bytes]=RESULT_MAGIC
    def serialize(self)->bytes:
        payload=b"".join(d.pack() for d in self.detections)
        return RESULT_HEADER.pack(self.magic,VERSION,RESULT_HEADER.size,self.stream_id,self.frame_id,self.timestamp_ns,len(self.detections),self.traffic_light_state,self.lane_state,self.adas_action,0,self.warning_flags,len(payload))+payload
    @classmethod
    def deserialize(cls,data:bytes):
        if len(data)<RESULT_HEADER.size: raise ProtocolError("truncated result header")
        m,v,hlen,sid,fid,ts,n,tls,lan,act,_r,warn,size=RESULT_HEADER.unpack_from(data)
        if m!=RESULT_MAGIC: raise ProtocolError("bad result magic")
        _check_version(v)
        if hlen!=RESULT_HEADER.size or size!=len(data)-hlen or size != n*DETECTION.size: raise ProtocolError("invalid result payload size")
        ds=tuple(DetectionRecord.unpack(data[hlen+i*DETECTION.size:hlen+(i+1)*DETECTION.size]) for i in range(n))
        return cls(sid,fid,ts,ds,tls,lan,act,warn)
