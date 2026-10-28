from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=p.read_text(encoding='utf-8')
s=s.replace('"AI sidecar 未连接 · 设计演示"','"AI sidecar 未连接"')
s=s.replace('text: "设计预览"','text: "未接入"')
p.write_text(s,encoding='utf-8')
print('done')
