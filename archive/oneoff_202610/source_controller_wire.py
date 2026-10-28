from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('#include "hud_control.h"','#include "hud_control.h"\n#include "source_controller.h"',1)
needle='''    view.rootContext()->setContextProperty(QStringLiteral("sceneEnabled"), sceneEnabled);'''
rep='''    view.rootContext()->setContextProperty(QStringLiteral("sceneEnabled"), sceneEnabled);
    auto *sourceSwitch = new SourceSwitchController(fps, aiFrameFile, aiHz,
                                                     pciePgvFile, localPgvFile,
                                                     view.rootContext());
    view.rootContext()->setContextProperty(QStringLiteral("sourceSwitch"), sourceSwitch);'''
if needle not in s: raise SystemExit('context marker')
s=s.replace(needle,rep,1)
needle='''    if (!video) {
        qCritical() << "QML VideoItem was not created";
        return 2;
    }'''
rep='''    if (!video) {
        qCritical() << "QML VideoItem was not created";
        return 2;
    }
    sourceSwitch->setVideo(video);'''
s=s.replace(needle,rep,1)
p.write_text(s,encoding='utf-8')
# QML button
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
            x: 112; y: 24; text: "切换"
            color: root.cyanHot; font.family: root.cjk; font.pixelSize: 6
        }
        MouseArea {
            anchors.fill: parent
            cursorShape: Qt.PointingHandCursor
            onClicked: sourceSwitch.toggle()
        }
    }'''
if needle not in s: raise SystemExit('chip marker')
s=s.replace(needle,rep,1)
p.write_text(s,encoding='utf-8')
# pro add files
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\qt_gpu_probe.pro')
s=p.read_text(encoding='utf-8')
s=s.replace('''    hud_control.cpp''','''    hud_control.cpp \\
    source_controller.cpp''',1)
s=s.replace('''    hud_control.h''','''    hud_control.h \\
    source_controller.h''',1)
p.write_text(s,encoding='utf-8')
print('controller wired')
