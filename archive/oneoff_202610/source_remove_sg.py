from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=p.read_text(encoding='utf-8')
s=s.replace('''            visible: root.viewMode !== 2 && !videoBackendSg
            sourceViewport: {''','''            visible: root.viewMode !== 2
            sourceViewport: {''',1)
start=s.find('''        VideoTextureItem {''')
if start>=0:
    end=s.find('''        Item {
            id: spatialSceneHost''', start)
    if end<0: raise SystemExit('spatial marker')
    s=s[:start]+s[end:]
p.write_text(s,encoding='utf-8')
print('removed SG QML block')
