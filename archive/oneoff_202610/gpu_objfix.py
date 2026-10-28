from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('#include <QDateTime>\n','#include <QCoreApplication>\n#include <QDateTime>\n')
s=s.replace('void carMesh(std::vector<float> &v, float x, float z, const QString &cls, bool ego) {','static QVector<ModelFace> *g_modelFaces = nullptr;\n\nvoid carMesh(std::vector<float> &v, float x, float z, const QString &cls, bool ego) {',1)
s=s.replace('''static QVector<ModelFace> *g_modelFaces = nullptr;

static QVector<ModelFace> loadModelFaces() {''','''static QVector<ModelFace> loadModelFaces() {''',1)
p.write_text(s,encoding='utf-8')
print('model pointer fixed')
