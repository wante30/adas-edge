from pathlib import Path
from collections import Counter
c=Counter(); m='?'
for ln in Path(r'C:\Users\JingyuWang\adas-edge\tools\competition_console\assets\source\car.obj').read_text(errors='ignore').splitlines():
 p=ln.split()
 if not p: continue
 if p[0]=='usemtl': m=p[1]
 elif p[0]=='f': c[m]+=1
print(c)
