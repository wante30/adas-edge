from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=p.read_text(encoding='utf-8')
# root function/reparent
s=s.replace('''    onViewModeChanged: if (video) video.spatialViewActive = (viewMode === 2)

    Component.onCompleted: {
        video.aiRoadQuadrant = roadQuadrant
        video.aiNavigationQuadrant = navQuadrant
    }''','''    function attachTeslaScene() {
        if (!teslaLive) return
        var target = (viewMode === 2) ? spatialSceneHost : cardSceneHost
        if (teslaLive.parent !== target) {
            teslaLive.parent = target
            teslaLive.anchors.fill = target
        }
        teslaLive.visible = (viewMode === 0 || viewMode === 2)
    }

    onViewModeChanged: {
        if (video) video.spatialViewActive = (viewMode === 2)
        attachTeslaScene()
    }

    Component.onCompleted: {
        video.aiRoadQuadrant = roadQuadrant
        video.aiNavigationQuadrant = navQuadrant
        attachTeslaScene()
    }''',1)
# central host
s=s.replace('''        TeslaBevView {
            id: teslaScene
            x: 0; y: 0; width: mainView.width; height: mainView.height
            visible: root.viewMode === 2
            compact: false
        }''','''        Item {
            id: spatialSceneHost
            x: 0; y: 0; width: mainView.width; height: mainView.height
            visible: root.viewMode === 2
        }''',1)
# card host + single scene item
s=s.replace('''        TeslaBevView {
            x: 14; y: 37; width: parent.width - 28; height: 88
            compact: true
        }''','''        Item {
            id: cardSceneHost
            x: 14; y: 37; width: parent.width - 28; height: 88
            visible: root.viewMode === 0
            TeslaSceneItem {
                id: teslaLive
                anchors.fill: parent
                boxes: video.aiBoxes
                online: video.aiOnline
                sourceKey: video.sourceMode
                sourceFrameId: video.displayFrameId
                sourceTimestampNs: video.displayTimestampNs
            }
        }''',1)
# card only in single; insert after layer.smooth
s=s.replace('''        layer.smooth: true; y: 547; width: 427; height: 145''','''        layer.smooth: true; y: 547; width: 427; height: 145
        visible: root.viewMode === 0''',1)
# labels/card badge
s=s.replace('''            text: "TESLA-BEVCAM-CLONE · 原版动态 GIF"''','''            text: "LIVE TRACKS · RELATIVE_ONLY"''')
s=s.replace('''            text: "DEMO · 非标定空间示意"''','''            text: "真实 Track ID · 非标定相对空间"''')
s=s.replace('''            color: "#2B220E"; border.color: "#6B552A"; border.width: 0.6
            Text { anchors.centerIn: parent; text: "DEMO"; color: root.amber; font.family: root.num; font.pixelSize: 6 }''','''            color: "#0B302A"; border.color: "#3F7059"; border.width: 0.6
            Text { anchors.centerIn: parent; text: "LIVE"; color: root.mint; font.family: root.num; font.pixelSize: 6 }''')
s=s.replace('''            text: "TESLA-BEVCAM-CLONE · 原版动态演示 · 非标定"''','''            text: "TESLA-BEVCAM-CLONE · LIVE · RELATIVE_ONLY"''')
p.write_text(s,encoding='utf-8')
print('shared live scene qml patched')
