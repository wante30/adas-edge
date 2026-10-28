from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=p.read_text(encoding='utf-8')
block='''        Text {
            x: 112; y: 24; text: "切换"
            color: root.cyanHot; font.family: root.cjk; font.pixelSize: 6
        }
        MouseArea {
            anchors.fill: parent
            cursorShape: Qt.PointingHandCursor
            onClicked: sourceSwitch.toggle()
        }
'''
if block not in s: raise SystemExit('switch block missing')
s=s.replace(block,'',1)
p.write_text(s,encoding='utf-8')
print('removed switch qml')
