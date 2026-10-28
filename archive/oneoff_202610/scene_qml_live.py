from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=p.read_text(encoding='utf-8')
# Replace central GIF host with spatial host
old='''        TeslaBevView {
            id: teslaScene
            x: 0; y: 0; width: mainView.width; height: mainView.height
            visible: root.viewMode === 2
            compact: false
        }
'''
new='''        Item {
            id: spatialSceneHost
            x: 0; y: 0; width: mainView.width; height: mainView.height
            visible: root.viewMode === 2
        }
'''
if old not in s: raise SystemExit('central host marker')
s=s.replace(old,new,1)
# Replace card host
old='''        TeslaBevView {
            x: 14; y: 37; width: parent.width - 28; height: 88
            compact: true
        }
'''
new='''        Item {
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
        }
'''
if old not in s: raise SystemExit('card host marker')
s=s.replace(old,new,1)
# card visible single only
s=s.replace('''        layer.smooth: true; y: 547; width: 427; height: 145''','''        layer.smooth: true; y: 547; width: 427; height: 145
        visible: root.viewMode === 0''',1)
# header labels update
s=s.replace('''            text: "TESLA-BEVCAM-CLONE · 原版动态 GIF"''','''            text: "LIVE TRACKS · RELATIVE_ONLY"''')
s=s.replace('''            text: "DEMO · 非标定空间示意"''','''            text: "真实 Track ID · 非标定相对空间"''')
s=s.replace('''            x: parent.width - 72; y: 9; width: 57; height: 16; radius: 2
            color: "#2B220E"; border.color: "#6B552A"; border.width: 0.6
            Text { anchors.centerIn: parent; text: "DEMO"; color: root.amber; font.family: root.num; font.pixelSize: 6 }''','''            x: parent.width - 72; y: 9; width: 57; height: 16; radius: 2
            color: "#0B302A"; border.color: "#3F7059"; border.width: 0.6
            Text { anchors.centerIn: parent; text: "LIVE"; color: root.mint; font.family: root.num; font.pixelSize: 6 }''')
s=s.replace('''            text: "TESLA-BEVCAM-CLONE · 原版动态演示 · 非标定"''','''            text: "TESLA-BEVCAM-CLONE · LIVE · RELATIVE_ONLY"''')
# add attach function near root existing onViewModeChanged
needle='''    onViewModeChanged: {
        if (video) video.spatialViewActive = (viewMode === 2)'''
if needle not in s:
    raise SystemExit('view mode marker')
s=s.replace(needle,'''    function attachTeslaScene() {
        if (!teslaLive) return
        var target = (viewMode === 2) ? spatialSceneHost : cardSceneHost
        if (teslaLive.parent !== target) {
            teslaLive.parent = target
            teslaLive.anchors.fill = target
        }
        teslaLive.visible = (viewMode === 0 || viewMode === 2)
    }

    onViewModeChanged: {
        attachTeslaScene()
        if (video) video.spatialViewActive = (viewMode === 2)''',1)
# Component.onCompleted add attach after existing video line
old='''        if (video) video.spatialViewActive = (viewMode === 2)
        if (video) video.aiRoadQuadrant = roadQuadrant'''
# find root completion occurrence maybe exact
if old in s:
    s=s.replace(old,old+'''
        attachTeslaScene()''',1)
else:
    # generic append to Component.onCompleted after first assignment
    comp=s.find('    Component.onCompleted: {')
    if comp>=0:
        pos=s.find('\n',comp)
        s=s[:pos+1]+'        attachTeslaScene()\n'+s[pos+1:]
p.write_text(s,encoding='utf-8')
print('single shared scene qml')
