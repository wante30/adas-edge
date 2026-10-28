from pathlib import Path
import numpy as np
vs=[]; fs=[]
for ln in Path(r'C:\Users\JingyuWang\adas-edge\tools\competition_console\assets\source\car.obj').read_text(errors='ignore').splitlines():
    p=ln.split()
    if not p: continue
    if p[0]=='v': vs.append([float(x) for x in p[1:4]])
    elif p[0]=='f':
        idx=[int(x.split('/')[0])-1 for x in p[1:]]
        if len(idx)>=3: fs.append(idx)
print('verts',len(vs),'faces',len(fs))
a=np.array(vs); print('min',a.min(0),'max',a.max(0),'mean',a.mean(0))
