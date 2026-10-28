from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('#include "source_controller.h"\n','',1)
s=s.replace('''    auto *sourceSwitch = new SourceSwitchController(fps, aiFrameFile, aiHz,
                                                     pciePgvFile, localPgvFile,
                                                     view.rootContext());
    view.rootContext()->setContextProperty(QStringLiteral("sourceSwitch"), sourceSwitch);
''','',1)
s=s.replace('''    sourceSwitch->setVideo(video);
''','',1)
p.write_text(s,encoding='utf-8')
print('removed controller from runtime')
