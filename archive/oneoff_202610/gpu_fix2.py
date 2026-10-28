from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
for block in ['''bool TeslaSceneItem::online() const { return online_.load(); }
''','''bool TeslaSceneItem::demoMode() const { return demo_mode_.load(); }
''']:
    s=s.replace(block,'')
p.write_text(s,encoding='utf-8')
print('dupes removed')
