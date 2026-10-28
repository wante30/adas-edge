from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=p.read_text(encoding='utf-8')
s=s.replace('''    property int trackingMode: 2  // 0=RAW, 1=FIX4, 2=TIME''','''    property int trackingMode: 2  // 0=RAW, 1=FIX4, 2=TIME
    property int sceneTick: 0''')
# set active on mode changes after existing onNavQuadrantChanged
s=s.replace('''    onNavQuadrantChanged: if (video) video.aiNavigationQuadrant = navQuadrant''','''    onNavQuadrantChanged: if (video) video.aiNavigationQuadrant = navQuadrant
    onViewModeChanged: if (video) video.spatialViewActive = (viewMode === 2)''')
old='''        AnimatedImage {
            x: 0; y: 0; width: 720; height: 460
            source: "assets/tesla_bev_demo.gif"
            fillMode: Image.PreserveAspectCrop
            visible: root.viewMode === 2
            playing: root.viewMode === 2
            cache: true
            smooth: true
        }'''
new='''        Image {
            id: teslaScene
            x: 0; y: 0; width: mainView.width; height: mainView.height
            source: "file:///dev/shm/tesla_scene_a.bmp"
            fillMode: Image.PreserveAspectFit
            visible: root.viewMode === 2
            cache: false
            smooth: true
            asynchronous: true
        }
        Timer {
            interval: 200; repeat: true; running: root.viewMode === 2
            onTriggered: {
                root.sceneTick = (root.sceneTick + 1) % 2
                teslaScene.source = root.sceneTick
                        ? "file:///dev/shm/tesla_scene_b.bmp"
                        : "file:///dev/shm/tesla_scene_a.bmp"
            }
        }'''
if old not in s: raise SystemExit('animated block')
s=s.replace(old,new)
s=s.replace('text: "BEV 视觉演示"','text: "BEV 空间场景"')
s=s.replace('text: "上游渲染序列 · 非实时几何"','text: "LIVE TRACKS · 5 Hz · 非实时几何"')
s=s.replace('text: "ADAS 计数来自右侧真实 AI"','text: "目标来自真实 Track ID · 坐标未标定"')
# initial active when completed
s=s.replace('''    Component.onCompleted: {
        if (video) video.aiRoadQuadrant = roadQuadrant''','''    Component.onCompleted: {
        if (video) video.spatialViewActive = (viewMode === 2)
        if (video) video.aiRoadQuadrant = roadQuadrant''')
p.write_text(s,encoding='utf-8')
print('qml patched')
