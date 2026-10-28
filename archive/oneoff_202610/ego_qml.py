from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=p.read_text(encoding='utf-8')
old='''        TeslaSceneItem {
            id: teslaScene
            x: 0; y: 0; width: mainView.width; height: mainView.height
            visible: root.viewMode === 2
            boxes: video.aiBoxes
            online: video.aiOnline
            sourceKey: video.sourceMode
            demoMode: root.sceneDemo
        }
'''
new='''        TeslaSceneItem {
            id: teslaScene
            x: 0; y: 0; width: mainView.width; height: mainView.height
            visible: root.viewMode === 2
            boxes: video.aiBoxes
            online: video.aiOnline
            sourceKey: video.sourceMode
            demoMode: root.sceneDemo
        }
        Image {
            id: egoOverlay
            visible: root.viewMode === 2
            source: "assets/ego_car_rear.png"
            width: 205
            height: width * 740.0 / 680.0
            x: (mainView.width - width) / 2
            y: mainView.height - height - 2
            fillMode: Image.PreserveAspectFit
            smooth: true
            cache: true
        }
'''
if old not in s: raise SystemExit('scene qml')
s=s.replace(old,new,1)
p.write_text(s,encoding='utf-8')
print('ego overlay qml')
