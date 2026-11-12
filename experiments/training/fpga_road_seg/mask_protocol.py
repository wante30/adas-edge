"""Offline-only Q4 mask envelope, never assigned an existing PCIe/HUD register.

Canonical CHW, row-major Q4-native coordinates; distinct from 1280x720 mosaic.
No board transport, device IO or command/control side effects in this module.
"""
from __future__ import annotations
import base64
import math
import zlib
import numpy as np
from . import CLASSES

MAX_PIXELS=1920*1080
ENCODINGS=('bitpack-le','rle-u32','probability-u8')


def shape_check(shape):
    if not isinstance(shape,(list,tuple)) or len(shape)!=3 or any(type(n) is not int for n in shape):
        raise ValueError('shape must be integer [4,height,width]')
    c,h,w=shape
    if c!=4 or h<1 or w<1 or h*w>MAX_PIXELS:
        raise ValueError('mask shape outside supported envelope')
    return c,h,w


def encode(mask, *, frame_id, timestamp_ns, encoding='bitpack-le', confidence=None,
           valid=True, stale=False, stream_epoch=0):
    a=np.asarray(mask)
    shape_check(list(a.shape))
    if not np.isfinite(a).all() or a.min()<0 or a.max()>1:
        raise ValueError('mask/probabilities must be finite in [0,1]')
    if encoding=='bitpack-le':
        # Each row independently padded to whole bytes; LSB is the leftmost pixel.
        payload=np.packbits(a>=.5,axis=-1,bitorder='little').tobytes()
    elif encoding=='probability-u8':
        payload=np.rint(a*255).astype(np.uint8).tobytes()
    elif encoding=='rle-u32':
        # Alternating zero/one runs on flattened CHW, always starting with zero.
        bits=(a>=.5).reshape(-1)
        cuts=np.flatnonzero(bits[1:]!=bits[:-1])+1
        counts=np.diff(np.concatenate(([0],cuts,[bits.size])))
        if bits[0]: counts=np.concatenate(([0],counts))
        payload=counts.astype('<u4').tobytes()
    else:
        raise ValueError('unsupported encoding')
    packet=dict(schema='Q4MASK-DRAFT-1',source_id='Q4',coordinate_space='q4_native',
                stream_epoch=stream_epoch,frame_id=frame_id,timestamp_ns=timestamp_ns,
                clock='capture_monotonic_ns',shape=list(a.shape),class_names=list(CLASSES),
                encoding=encoding,confidence=confidence if confidence is not None else [None]*4,
                valid=valid,stale=stale,hardware=False,provider='offline_mock',
                payload_bytes=len(payload),crc32=f'{zlib.crc32(payload):08x}',
                payload_base64=base64.b64encode(payload).decode('ascii'))
    # Validate exactly the same header a receiver will see.
    decode(packet)
    return packet


def decode(packet):
    if packet.get('schema')!='Q4MASK-DRAFT-1' or packet.get('source_id')!='Q4' or packet.get('coordinate_space')!='q4_native':
        raise ValueError('only Q4-native draft packets are accepted')
    if packet.get('class_names')!=list(CLASSES) or packet.get('clock')!='capture_monotonic_ns':
        raise ValueError('class order/clock mismatch')
    for key in ('frame_id','timestamp_ns','stream_epoch','payload_bytes'):
        if type(packet.get(key)) is not int or packet[key]<0:
            raise ValueError(f'nonnegative integer {key} required')
    for key in ('valid','stale','hardware'):
        if type(packet.get(key)) is not bool:
            raise ValueError(f'boolean {key} required')
    if packet['hardware'] or packet.get('provider')!='offline_mock':
        raise ValueError('this implementation is offline/mock only')
    conf=packet.get('confidence')
    if not isinstance(conf,list) or len(conf)!=4 or any(x is not None and (type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1) for x in conf):
        raise ValueError('confidence must be four probabilities or null')
    c,h,w=shape_check(packet.get('shape'))
    encoding=packet.get('encoding')
    if encoding not in ENCODINGS:
        raise ValueError('unsupported encoding')
    size={'bitpack-le':c*h*((w+7)//8),'probability-u8':c*h*w}.get(encoding)
    # Bound RLE before base64 allocation: worst case alternating bits plus zero prefix.
    maximum=size if size is not None else 4*(c*h*w+1)
    if packet['payload_bytes']>maximum or not isinstance(packet.get('payload_base64'),str) or len(packet['payload_base64'])>4*((maximum+2)//3):
        raise ValueError('oversized payload')
    try:
        payload=base64.b64decode(packet['payload_base64'],validate=True)
    except Exception as e:
        raise ValueError('invalid base64 payload') from e
    if len(payload)!=packet['payload_bytes'] or f'{zlib.crc32(payload):08x}'!=packet.get('crc32'):
        raise ValueError('length/CRC mismatch')
    if size is not None and len(payload)!=size:
        raise ValueError('shape/payload size mismatch')
    if encoding=='bitpack-le':
        data=np.frombuffer(payload,np.uint8).reshape(c,h,(w+7)//8)
        # Nonzero unused row bits are forbidden, so serialization is canonical.
        if w%8 and np.any(data[:,:,-1] >> (w%8)):
            raise ValueError('nonzero bitpack padding')
        return np.unpackbits(data,axis=-1,count=w,bitorder='little').astype(np.float32)
    if encoding=='probability-u8':
        return np.frombuffer(payload,np.uint8).reshape(c,h,w).astype(np.float32)/255
    if len(payload)%4 or not payload:
        raise ValueError('invalid RLE u32 payload')
    runs=np.frombuffer(payload,dtype='<u4')
    if int(runs.astype(np.uint64).sum())!=c*h*w or np.any(runs[1:]==0):
        raise ValueError('RLE counts do not cover exactly the mask')
    return np.repeat(np.arange(len(runs),dtype=np.uint32)%2,runs).reshape(c,h,w).astype(np.float32)


class OfflineMaskProvider:
    def __init__(self, packet, max_age_ms=150):
        self.mask=decode(packet); self.packet=packet
        if not math.isfinite(max_age_ms) or max_age_ms<0:
            raise ValueError('max_age_ms must be finite and nonnegative')
        self.max_age_ns=int(max_age_ms*1e6)

    def get(self, *, frame_id, timestamp_ns, now_ns, shape, stream_epoch=0):
        p=self.packet
        identity=(p['frame_id']==frame_id and p['timestamp_ns']==timestamp_ns and
                  p['stream_epoch']==stream_epoch and list(self.mask.shape[1:])==list(shape))
        fresh=0<=now_ns-p['timestamp_ns']<=self.max_age_ns
        valid=bool(p['valid'] and not p['stale'] and identity and fresh)
        status=dict(valid=valid,stale=not fresh or p['stale'],identity_matches=identity,
                    age_ms=(now_ns-p['timestamp_ns'])/1e6,hardware=False,provider='offline_mock')
        return (self.mask if valid else None),status


def transfer_budget(width=320,height=192,fps=30):
    if width<1 or height<1 or fps<=0: raise ValueError('positive geometry/rate required')
    raw=4*width*height
    sizes={'bitpack_le_bytes':4*height*((width+7)//8),'probability_u8_bytes':raw,
           'rle_best_case_bytes':4,'rle_worst_case_bytes':4*(raw+1)}
    return dict(width=width,height=height,fps=fps,payload_only=sizes,
                bytes_per_second={k:v*fps for k,v in sizes.items()},
                note='Excludes envelope/base64/transport framing/CRC/PCIe transactions; RLE content dependent.')
