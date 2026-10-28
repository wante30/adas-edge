from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=p.read_text(encoding='utf-8')
# root property remove
s=s.replace('''    property bool sceneDemo: false
''','')
# replace central GPU item + ego overlay
start=s.index('        TeslaSceneItem {')
end=s.index('        // Readability gradients.', start)
new='''        TeslaBevView {
            id: teslaScene
            x: 0; y: 0; width: mainView.width; height: mainView.height
            visible: root.viewMode === 2
            compact: false
        }

'''
s=s[:start]+new+s[end:]
# remove scene mode chip
chip_start=s.find('''        Rectangle {
            x: parent.width - 300; y: 44; width: 88; height: 23; radius: 2''')
if chip_start >= 0:
    chip_end=s.index('\n        Rectangle {\n            x: parent.width - 202; y: 44', chip_start)
    s=s[:chip_start]+s[chip_end+1:]
# replace spatial bottom card
start=s.index('    Item {\n        id: spatialCard')
end=s.index('    // ------------------------------------------------------------ signal test', start)
new_card='''    Item {
        id: spatialCard
        x: 833
        layer.enabled: true
        layer.smooth: true; y: 547; width: 427; height: 145
        HudPanel {
            anchors.fill: parent
            cut: 11; accent: root.cyan
            fillTop: "#091D27"; fillBottom: "#040F16"
            fillOpacity: 0.96; borderOpacity: 0.58
        }
        Image { x: 14; y: 11; width: 15; height: 15; source: "assets/icon_radar.png" }
        Text {
            x: 35; y: 7; text: "Tesla BEV 空间"
            color: root.ink; font.family: root.cjk; font.pixelSize: 10; font.weight: Font.DemiBold
        }
        Rectangle {
            x: parent.width - 72; y: 9; width: 57; height: 16; radius: 2
            color: "#2B220E"; border.color: "#6B552A"; border.width: 0.6
            Text { anchors.centerIn: parent; text: "DEMO"; color: root.amber; font.family: root.num; font.pixelSize: 6 }
        }
        Rectangle { x: 14; y: 31; width: parent.width - 28; height: 1; color: "#1A3942" }
        TeslaBevView {
            x: 14; y: 37; width: parent.width - 28; height: 88
            compact: true
        }
        Text {
            x: 16; y: parent.height - 15
            text: "TESLA-BEVCAM-CLONE · 原版动态演示 · 非标定"
            color: "#7FB3B8"; font.family: root.cjk; font.pixelSize: 6
        }
    }

'''
s=s[:start]+new_card+s[end:]
p.write_text(s,encoding='utf-8')
print('restored original Tesla card')
