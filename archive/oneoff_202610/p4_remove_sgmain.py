from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('#include "video_texture_item.h"\n','',1)
s=s.replace('''    qmlRegisterType<VideoTextureItem>("QtGpuProbe", 1, 0, "VideoTextureItem");\n''','',1)
s=s.replace('''    view.rootContext()->setContextProperty(QStringLiteral("videoBackendSg"),
                                             qgetenv("QT_VIDEO_RENDER_BACKEND") == "sg");
''','',1)
p.write_text(s,encoding='utf-8')
print('removed SG registration')
