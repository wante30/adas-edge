"""96-bit AVM LUT encoder and fast PC replay of the existing RTL contract."""
from pathlib import Path
import numpy as np

FIELDS = dict(addr_a=(0,18), addr_b=(18,18), fx_a=(36,8), fy_a=(44,8),
              fx_b=(52,8), fy_b=(60,8), alpha_q=(68,11), cam_a=(79,3),
              cam_b=(82,3), valid=(85,1), blend=(86,1))


def validate(fields,source_size=(640,320)):
    sw,sh=source_size
    valid=np.asarray(fields['valid'],bool);mix=np.asarray(fields['blend'],bool)
    if np.any(mix & ~valid) or np.any(np.asarray(fields['alpha_q']) > 64):
        raise ValueError('Invalid blend control')
    if np.any(valid & ~mix & (np.asarray(fields['alpha_q']) != 64)):
        raise ValueError('Single source must have alpha 64')
    for side,active in [('a',valid),('b',valid & mix)]:
        addr=np.asarray(fields[f'addr_{side}'])
        if np.any(active & ((addr%sw >= sw-1)|(addr//sw >= sh-1))):
            raise ValueError('Invalid source neighbourhood')
        if np.any(active & (np.asarray(fields[f'cam_{side}']) >= 4)):
            raise ValueError('Invalid camera slot')
        for axis in ['fx','fy']:
            if np.any(np.asarray(fields[f'{axis}_{side}']) > 15):
                raise ValueError('Fraction exceeds Q0.4')


def encode(fields):
    validate(fields)
    n = np.asarray(fields['valid']).size
    raw = np.zeros((n,12), np.uint8)
    for key, (bit,width) in FIELDS.items():
        a = np.asarray(fields[key]).reshape(-1)
        if a.size != n or np.any(a < 0) or np.any(a >= 1 << width) or np.any(a != np.floor(a)):
            raise ValueError(f'Invalid {key}')
        a = a.astype(np.uint64) << (bit % 8)
        for j in range((bit % 8 + width + 7)//8):
            raw[:,bit//8+j] |= ((a >> (8*j)) & 255).astype(np.uint8)
    return raw


def decode(raw):
    raw = np.asarray(raw, np.uint8).reshape(-1,12)
    if np.any(raw[:,11]) or np.any(raw[:,10] & 128):
        raise ValueError('Nonzero reserved bits')
    fields = {}
    for key,(bit,width) in FIELDS.items():
        v = np.zeros(len(raw),np.uint64)
        for j in range((bit % 8 + width + 7)//8):
            v |= raw[:,bit//8+j].astype(np.uint64) << (8*j)
        fields[key] = ((v >> (bit % 8)) & ((1 << width)-1)).astype(np.uint32)
    validate(fields)
    return fields


def from_maps(maps, ca, cb, alpha, valid, blend, source_size=(640,320)):
    sw,sh = source_size
    if sw*sh > (1 << 18):
        raise ValueError('Source pixels exceed 18-bit LUT address space')
    if not np.all(np.isfinite(alpha)) or np.any((alpha < 0)|(alpha > 1)):
        raise ValueError('Alpha must be finite and within [0,1]')
    if np.any(valid & ((ca < 0)|(ca > 3))) or np.any(valid & blend & ((cb < 0)|(cb > 3))):
        raise ValueError('Invalid camera slot')
    fields = dict(valid=valid.ravel(), blend=(valid & blend).ravel())
    for side,cam,active in [('a',ca,valid),('b',cb,valid & blend)]:
        uv = np.take_along_axis(maps,cam[None,...,None],axis=0)[0]
        q = np.floor(np.where(active[...,None],uv,0)*16+.5).astype(np.int64)
        x,y = q[...,0]//16,q[...,1]//16
        if np.any(active & ((x < 0)|(x >= sw-1)|(y < 0)|(y >= sh-1))):
            raise ValueError('Bilinear neighbourhood outside input frame')
        fields[f'addr_{side}'] = (y*sw+x).ravel()
        fields[f'fx_{side}'] = (q[...,0]%16).ravel()
        fields[f'fy_{side}'] = (q[...,1]%16).ravel()
        fields[f'cam_{side}'] = np.where(active,cam,0).ravel()
    fields['alpha_q'] = np.where(valid,np.where(blend,np.floor(alpha*64+.5),64),0).ravel()
    if np.any(fields['alpha_q'] > 64):
        raise ValueError('Blend alpha exceeds unity')
    return fields


def write_lut(folder: Path, fields):
    folder.mkdir(exist_ok=True,parents=True)
    raw = encode(fields)
    raw.tofile(folder/'avm_slam_96bit_le.bin')
    with (folder/'avm_slam_96bit.hex').open('w',encoding='ascii',newline='\n') as f:
        for row in raw:
            f.write(row[::-1].tobytes().hex()+'\n')
    return decode(raw)


def pack565(bgr):
    b,g,r = [bgr[...,i].astype(np.uint16) for i in range(3)]
    return ((r>>3)<<11)|((g>>2)<<5)|(b>>3)


def expand565(words):
    b = words & 31; g = (words >> 5) & 63; r = (words >> 11) & 31
    return np.stack(((b<<3)|(b>>2),(g<<2)|(g>>4),(r<<3)|(r>>2)),-1).astype(np.uint8)


class Replay:
    """Vectorized integer replay, using only decoded LUT fields.

    Do not substitute OpenCV remap: interpolation rounding can differ by a
    code value at ties even for exactly representable Q0.4 coordinates.
    """
    def __init__(self,fields,size=(640,640),source_size=(640,320)):
        self.f = {k:v.reshape(size[1],size[0]) for k,v in fields.items()}
        f=self.f; sw,sh=source_size
        self.indices=[]; self.weights=[]
        for side in ['a','b']:
            active=(f['valid']!=0) & ((f['blend']!=0) if side=='b' else True)
            addr=np.where(active,f[f'addr_{side}']+f[f'cam_{side}']*sw*sh,0)
            self.indices.append(np.stack([addr,addr+1,addr+sw,addr+sw+1]))
            fx=f[f'fx_{side}'];fy=f[f'fy_{side}']
            self.weights.append(np.stack([(16-fx)*(16-fy),fx*(16-fy),(16-fx)*fy,fx*fy]))

    def render(self,words):
        f=self.f
        flat=np.stack(words).reshape(-1)
        samples=[flat[idx].astype(np.uint32) for idx in self.indices]
        channels=[]
        for shift,bits in [(0,5),(5,6),(11,5)]:
            sides=[]
            for sampled,w in zip(samples,self.weights):
                v=(sampled>>shift)&((1<<bits)-1)
                v=(v<<(8-bits))|(v>>(2*bits-8))
                sides.append(((v*w).sum(axis=0,dtype=np.uint32)+128)>>8)
            alpha=f['alpha_q']
            channels.append(((alpha*sides[0]+(64-alpha)*sides[1]+32)>>6).astype(np.uint8))
        bgr=np.stack(channels,-1)
        result=pack565(bgr); result[f['valid']==0]=0
        return result
