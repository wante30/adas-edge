from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\ai_bridge.py')
s=p.read_text(encoding='utf-8')
if 'import signal' not in s:
    s=s.replace('import queue\n','import queue\nimport signal\n',1)
p.write_text(s,encoding='utf-8')
print('signal' in s)
