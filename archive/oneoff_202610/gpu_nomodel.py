from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''    if (!person && g_modelFaces) {
        modelCar(v, *g_modelFaces, x, z, d / 4.93f, w / 2.11f, ego);
        return;
    }
''','')
s=s.replace('''        if (!g_modelFaces) {
            g_modelFaces = new QVector<ModelFace>();
            *g_modelFaces = loadModelFaces();
        }
''','')
p.write_text(s,encoding='utf-8')
print('obj branch disabled')
