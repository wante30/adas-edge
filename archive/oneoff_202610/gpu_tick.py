from pathlib import Path
h=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.h')
s=h.read_text(encoding='utf-8').replace('    double sceneFps() const { return scene_fps_.load(); }','    double sceneFps() const { return scene_fps_.load(); }\n    Q_INVOKABLE void requestUpdate() { update(); }')
h.write_text(s,encoding='utf-8')
q=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=q.read_text(encoding='utf-8')
needle='''            sourceKey: video.sourceMode
            demoMode: root.sceneDemo
        }
'''
rep='''            sourceKey: video.sourceMode
            demoMode: root.sceneDemo
            Timer {
                interval: 16; repeat: true; running: root.viewMode === 2
                onTriggered: teslaScene.requestUpdate()
            }
        }
'''
if needle not in s: raise SystemExit('scene qml marker')
s=s.replace(needle,rep,1)
q.write_text(s,encoding='utf-8')
print('external scene tick')
