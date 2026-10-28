from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''    const bool autoCycleCapture = args.contains("--capture-cycle");''','''    const bool autoCycleCapture = args.contains("--capture-cycle");
    const bool sceneEnabled = !args.contains("--scene-off");''',1)
s=s.replace('''    view.rootContext()->setContextProperty(QStringLiteral("autoCycleCapture"), autoCycleCapture);''','''    view.rootContext()->setContextProperty(QStringLiteral("autoCycleCapture"), autoCycleCapture);
    view.rootContext()->setContextProperty(QStringLiteral("sceneEnabled"), sceneEnabled);''',1)
p.write_text(s,encoding='utf-8')
q=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=q.read_text(encoding='utf-8')
s=s.replace('''    property bool sceneDemo: false''','''    property bool sceneDemo: false''') if False else s
# add root sceneEnabled property after autoHudSend
s=s.replace('''    property bool autoHudSend: true''','''    property bool autoHudSend: true
    property bool sceneEnabled: (typeof sceneEnabled !== "undefined" ? sceneEnabled : true)''',1)
s=s.replace('''                boxes: video.aiBoxes
                online: video.aiOnline
                sourceKey: video.sourceMode''','''                boxes: video.aiBoxes
                online: video.aiOnline
                visible: root.sceneEnabled
                sourceKey: video.sourceMode''',1)
q.write_text(s,encoding='utf-8')
print('scene-off flag added')
