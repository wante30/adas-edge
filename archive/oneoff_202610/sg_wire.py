from pathlib import Path
repo=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe')
# main includes/register/context
p=repo/'main.cpp'; s=p.read_text(encoding='utf-8')
s=s.replace('#include "tesla_scene_item.h"','#include "tesla_scene_item.h"\n#include "video_texture_item.h"',1)
s=s.replace('''    qmlRegisterType<VideoItem>("QtGpuProbe", 1, 0, "VideoItem");''','''    qmlRegisterType<VideoItem>("QtGpuProbe", 1, 0, "VideoItem");
    qmlRegisterType<VideoTextureItem>("QtGpuProbe", 1, 0, "VideoTextureItem");''',1)
s=s.replace('''    view.rootContext()->setContextProperty(QStringLiteral("sceneEnabled"), sceneEnabled);''','''    view.rootContext()->setContextProperty(QStringLiteral("sceneEnabled"), sceneEnabled);
    view.rootContext()->setContextProperty(QStringLiteral("videoBackendSg"),
                                             qgetenv("QT_VIDEO_RENDER_BACKEND") == "sg");''',1)
p.write_text(s,encoding='utf-8')
# qml
p=repo/'main.qml'; s=p.read_text(encoding='utf-8')
s=s.replace('''            visible: root.viewMode !== 2
            sourceViewport: {''','''            visible: root.viewMode !== 2 && !videoBackendSg
            sourceViewport: {''',1)
needle='''        Item {
            id: spatialSceneHost'''
sg='''        VideoTextureItem {
            id: videoSg
            objectName: "videoTextureItem"
            source: video
            x: video.x; y: video.y
            width: video.width; height: video.height
            visible: root.viewMode !== 2 && videoBackendSg
        }

        Connections {
            target: video
            function onFrameAdvanced() { videoSg.requestUpdate() }
        }

        Item {
            id: spatialSceneHost'''
if needle not in s: raise SystemExit('qml spatial marker')
s=s.replace(needle,sg,1)
p.write_text(s,encoding='utf-8')
# pro
p=repo/'qt_gpu_probe.pro'; s=p.read_text(encoding='utf-8')
s=s.replace('''    video_item.cpp \\''','''    video_item.cpp \\
    video_texture_item.cpp \\''',1)
s=s.replace('''    video_item.h \\''','''    video_item.h \\
    video_texture_item.h \\''',1)
p.write_text(s,encoding='utf-8')
print('wired SG backend')
