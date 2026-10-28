from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=p.read_text(encoding='utf-8')
s=s.replace('''    property int sceneTick: 0
    property int sceneBuffer: 0
    property int scenePending: -1''','''    property bool sceneDemo: false''')
start=s.index('        Image {\n            id: sceneA')
end=s.index('        // Readability gradients.', start)
new='''        TeslaSceneItem {
            id: teslaScene
            x: 0; y: 0; width: mainView.width; height: mainView.height
            visible: root.viewMode === 2
            boxes: video.aiBoxes
            online: video.aiOnline
            sourceKey: video.sourceMode
            demoMode: root.sceneDemo
        }

'''
s=s[:start]+new+s[end:]
# add mode chip before TRK TIME block
marker='''        Rectangle {
            x: parent.width - 202; y: 44; width: 84; height: 23; radius: 2
            color: "#C0031118"; border.color: root.trackingMode === 2 ? "#3F7059" : "#5A4A2C"'''
chip='''        Rectangle {
            x: parent.width - 300; y: 44; width: 88; height: 23; radius: 2
            color: "#C0031118"; border.color: root.sceneDemo ? "#5A4A2C" : "#3F7059"; border.width: 0.7
            Text {
                anchors.centerIn: parent
                text: root.sceneDemo ? "SCENE DEMO" : "SCENE LIVE"
                color: root.sceneDemo ? root.amber : root.mint
                font.family: root.num; font.pixelSize: 6
            }
            MouseArea {
                anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                onClicked: root.sceneDemo = !root.sceneDemo
            }
        }

'''
if marker not in s: raise SystemExit('trk marker missing')
s=s.replace(marker,chip+marker,1)
p.write_text(s,encoding='utf-8')
print('qml gpu scene installed')
