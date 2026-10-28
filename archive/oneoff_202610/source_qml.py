from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=p.read_text(encoding='utf-8')
needle='''        Rectangle {
            x: 198; y: 14; width: 5; height: 5; radius: 2.5
            color: video.sourceOnline ? root.mint : root.amber
        }
    }'''
rep='''        Rectangle {
            x: 198; y: 14; width: 5; height: 5; radius: 2.5
            color: video.sourceOnline ? root.mint : root.amber
        }
        Text {
            x: 132; y: 24; text: "切换"
            color: root.cyanHot; font.family: root.cjk; font.pixelSize: 6
        }
        MouseArea {
            anchors.fill: parent
            cursorShape: Qt.PointingHandCursor
            onClicked: {
                var toPcie = video.sourceMode !== "pgv-live"
                video.switchPgvSource(toPcie ? pciePgvFile : localPgvFile,
                                      toPcie ? "pgv-live" : "pgv-local")
            }
        }
    }'''
if needle not in s: raise SystemExit('source chip marker')
s=s.replace(needle,rep,1)
p.write_text(s,encoding='utf-8')
print('source chip toggle')
