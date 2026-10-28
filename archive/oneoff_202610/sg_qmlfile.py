from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''    view.setSource(QUrl(QStringLiteral("qrc:/main.qml")));''','''    const QString qmlOverride = qEnvironmentVariable("QT_QML_FILE");
    view.setSource(qmlOverride.isEmpty() ? QUrl(QStringLiteral("qrc:/main.qml"))
                                         : QUrl::fromLocalFile(qmlOverride));''',1)
p.write_text(s,encoding='utf-8')
print('qml override')
