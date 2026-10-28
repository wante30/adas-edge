from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=p.read_text(encoding='utf-8')
s=s.replace('''            source: video
            x: video.x; y: video.y
            width: video.width; height: video.height
            visible: root.viewMode !== 2 && videoBackendSg''','''            source: video
            x: 0; y: (mainView.height - height) / 2
            width: mainView.width
            height: width * 9.0 / 16.0
            visible: root.viewMode !== 2 && videoBackendSg''',1)
p.write_text(s,encoding='utf-8')
print('explicit sg geometry')
