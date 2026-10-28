from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=p.read_text(encoding='utf-8')
s=s.replace('''    property int sceneTick: 0''','''    property int sceneTick: 0
    property int sceneBuffer: 0
    property int scenePending: -1''')
old='''        Image {
            id: teslaScene
            x: 0; y: 0; width: mainView.width; height: mainView.height
            source: "file:///dev/shm/tesla_scene_a.png"
            fillMode: Image.PreserveAspectFit
            visible: root.viewMode === 2
            cache: false
            smooth: true
            asynchronous: true
        }
        Timer {
            interval: 200; repeat: true; running: root.viewMode === 2
            onTriggered: {
                root.sceneTick = (root.sceneTick + 1) % 100000
                teslaScene.source = "file:///dev/shm/tesla_scene_a.png?t=" + root.sceneTick
            }
        }'''
new='''        Image {
            id: sceneA
            x: 0; y: 0; width: mainView.width; height: mainView.height
            source: "file:///dev/shm/tesla_scene_a.png"
            fillMode: Image.PreserveAspectFit
            visible: root.viewMode === 2 && root.sceneBuffer === 0
            cache: false
            smooth: true
            asynchronous: true
            onStatusChanged: if (status === Image.Ready && root.scenePending === 0) {
                root.sceneBuffer = 0; root.scenePending = -1
            }
        }
        Image {
            id: sceneB
            x: 0; y: 0; width: mainView.width; height: mainView.height
            source: "file:///dev/shm/tesla_scene_b.png"
            fillMode: Image.PreserveAspectFit
            visible: root.viewMode === 2 && root.sceneBuffer === 1
            cache: false
            smooth: true
            asynchronous: true
            onStatusChanged: if (status === Image.Ready && root.scenePending === 1) {
                root.sceneBuffer = 1; root.scenePending = -1
            }
        }
        Timer {
            interval: 200; repeat: true; running: root.viewMode === 2
            onTriggered: {
                if (root.scenePending !== -1) return
                var next = 1 - root.sceneBuffer
                root.sceneTick = (root.sceneTick + 1) % 100000
                root.scenePending = next
                if (next === 0)
                    sceneA.source = "file:///dev/shm/tesla_scene_a.png?t=" + root.sceneTick
                else
                    sceneB.source = "file:///dev/shm/tesla_scene_b.png?t=" + root.sceneTick
            }
        }'''
if old not in s: raise SystemExit('scene block')
s=s.replace(old,new)
p.write_text(s,encoding='utf-8')
print('dual buffer qml')
