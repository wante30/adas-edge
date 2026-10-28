from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('#include <QDebug>\n','#include <QDebug>\n#include <QFile>\n#include <QTextStream>\n#include <QVector3D>\n')
# insert model structs before carMesh
marker='void carMesh(std::vector<float> &v, float x, float z, const QString &cls, bool ego) {'
model='''struct ModelFace {
    QVector3D a, b, c;
    QVector3D n;
    int material = 0;
};

QVector3D materialColor(int material, bool ego) {
    switch (material) {
    case 1: return QVector3D(.035f, .040f, .045f);  // tire
    case 2: return QVector3D(.26f, .28f, .30f);     // metal
    case 3: return QVector3D(.06f, .10f, .13f);     // glass
    case 4: return QVector3D(.42f, .43f, .45f);     // trim
    case 5: return QVector3D(.72f, .62f, .34f);     // lamp
    default: return ego ? QVector3D(.20f, .29f, .39f) : QVector3D(.40f, .43f, .46f);
    }
}

void modelCar(std::vector<float> &v, const QVector<ModelFace> &faces, float x, float z,
              float scale, bool ego) {
    quad(v, x - 1.0f, z - 2.2f, x + 1.0f, z + 2.2f, .012f, .022f, .027f, .032f);
    const float cx = 1.428f, cz = .029f;
    const int step = ego ? 1 : 2;
    for (int i = 0; i < faces.size(); i += step) {
        const ModelFace &f = faces[i];
        QVector3D col = materialColor(f.material, ego);
        const float shade = .62f + .38f * std::max(0.0f, f.n.y());
        col *= shade;
        const float xs[3] = {f.a.x(), f.b.x(), f.c.x()};
        const float ys[3] = {f.a.y(), f.b.y(), f.c.y()};
        const float zs[3] = {f.a.z(), f.b.z(), f.c.z()};
        for (int k = 0; k < 3; ++k) {
            const float wx = x + (zs[k] - cz) * scale;
            const float wz = z + (xs[k] - cx) * scale;
            pushVertex(v, wx, std::max(0.015f, ys[k] * scale), wz, col.x(), col.y(), col.z());
        }
    }
}

'''
if marker not in s: raise SystemExit('car marker')
s=s.replace(marker, model+marker,1)
# branch in carMesh after dims/colors, before shadow
needle='''    // soft ground shadow
    quad(v, x - w * .62f, z - d * .55f, x + w * .62f, z + d * .55f, .012f, .025f, .03f, .035f);
'''
replacement='''    if (!truck && !bus && !person && _modelFaces) {
        const float scale = d / 4.93f;
        modelCar(v, *_modelFaces, x, z, scale, ego);
        return;
    }
    // soft ground shadow
    quad(v, x - w * .62f, z - d * .55f, x + w * .62f, z + d * .55f, .012f, .025f, .03f, .035f);
'''
if needle not in s: raise SystemExit('shadow marker')
s=s.replace(needle,replacement,1)
# Add static model pointer + loader before namespace end? insert before } // namespace
insert='''
static QVector<ModelFace> *g_modelFaces = nullptr;

static QVector<ModelFace> loadModelFaces() {
    QVector<ModelFace> faces;
    QFile file(QCoreApplication::applicationDirPath() + QStringLiteral("/assets/car.obj"));
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "car.obj unavailable:" << file.fileName();
        return faces;
    }
    QVector<QVector3D> vertices;
    QVector<int> faceVertices;
    int material = 0;
    QTextStream in(&file);
    while (!in.atEnd()) {
        const QString line = in.readLine().trimmed();
        if (line.isEmpty() || line.startsWith('#')) continue;
        const QStringList parts = line.split(' ', Qt::SkipEmptyParts);
        if (parts.size() < 2) continue;
        if (parts[0] == "v" && parts.size() >= 4)
            vertices.push_back(QVector3D(parts[1].toFloat(), parts[2].toFloat(), parts[3].toFloat()));
        else if (parts[0] == "usemtl") {
            const QString n = parts[1];
            if (n == "tire" || n == "Black_Plastic" || n == "tirehouse") material = 1;
            else if (n == "metal" || n == "metal2") material = 2;
            else if (n == "glass3") material = 3;
            else if (n == "plt1" || n == "plt3" || n == "plt4") material = 4;
            else if (n == "wire_115115115" || n == "wire_086086086") material = 5;
            else material = 0;
        } else if (parts[0] == "f") {
            faceVertices.clear();
            for (int i = 1; i < parts.size(); ++i) {
                const QString idx = parts[i].section('/', 0, 0);
                const int vi = idx.toInt() - 1;
                if (vi >= 0 && vi < vertices.size()) faceVertices.push_back(vi);
            }
            for (int i = 1; i + 1 < faceVertices.size(); ++i) {
                ModelFace f;
                f.a = vertices[faceVertices[0]];
                f.b = vertices[faceVertices[i]];
                f.c = vertices[faceVertices[i + 1]];
                f.n = QVector3D::crossProduct(f.b - f.a, f.c - f.a).normalized();
                f.material = material;
                faces.push_back(f);
            }
        }
    }
    qInfo() << "car.obj loaded faces:" << faces.size();
    return faces;
}
'''
if 'static QVector<ModelFace> *g_modelFaces' in s: raise SystemExit('loader exists')
# insert before unnamed namespace close first occurrence
pos=s.index('} // namespace')
s=s[:pos]+insert+s[pos:]
# replace _modelFaces pointer in carMesh with global, and init in renderer initGL
s=s.replace('if (!truck && !bus && !person && _modelFaces) {\n        const float scale = d / 4.93f;\n        modelCar(v, *_modelFaces, x, z, scale, ego);','if (!truck && !bus && !person && g_modelFaces) {\n        const float scale = d / 4.93f;\n        modelCar(v, *g_modelFaces, x, z, scale, ego);')
s=s.replace('''    void initGL() {
        program_ = std::make_unique<QOpenGLShaderProgram>();''','''    void initGL() {
        if (!g_modelFaces) {
            g_modelFaces = new QVector<ModelFace>();
            *g_modelFaces = loadModelFaces();
        }
        program_ = std::make_unique<QOpenGLShaderProgram>();''')
p.write_text(s,encoding='utf-8')
print('obj model integrated')
