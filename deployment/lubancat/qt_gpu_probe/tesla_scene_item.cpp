#include "tesla_scene_item.h"

#include <QCoreApplication>
#include <QDateTime>
#include <cstdio>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include <QVector3D>
#include <QElapsedTimer>
#include <QHash>
#include <QMatrix4x4>
#include <QOpenGLBuffer>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QImage>
#include <QOpenGLVertexArrayObject>
#include <QTimer>
#include <QVariantMap>
#include <QtMath>
#include <algorithm>
#include <memory>
#include <cmath>
#include <vector>

namespace {
struct SourceBox {
    int id = 0;
    QString cls;
    double x = 0, y = 0, w = 0, h = 0;
    double confidence = 0;
    int missed = 0;
    bool predicted = false;
};

struct LiveCar {
    float x = 0, z = 0;
    QString cls = QStringLiteral("car");
    double seen = 0;
};

struct DemoCar {
    float x = 0, z = 0;
    QString cls = QStringLiteral("car");
    float phase = 0;
};

struct Billboard {
    float x = 0, z = 0, w = 1.9f, h = 2.0f;
    QVector3D tint{0.55f, 0.60f, 0.65f};
};

void pushVertex(std::vector<float> &out, float x, float y, float z, float r, float g, float b) {
    out.insert(out.end(), {x, y, z, r, g, b});
}

void quad(std::vector<float> &v, float x1, float z1, float x2, float z2,
          float y, float r, float g, float b) {
    // x1/z1 and x2/z2 are opposite corners; build the four actual corners.
    const float xa = x1, za = z1, xb = x2, zb = z2;
    pushVertex(v, xa, y, za, r, g, b); pushVertex(v, xb, y, za, r, g, b);
    pushVertex(v, xb, y, zb, r, g, b);
    pushVertex(v, xa, y, za, r, g, b); pushVertex(v, xb, y, zb, r, g, b);
    pushVertex(v, xa, y, zb, r, g, b);
}

void box3(std::vector<float> &v, float cx, float y0, float cz, float w, float h, float d,
          float r, float g, float b, bool darkBottom = true) {
    const float x0 = cx - w * .5f, x1 = cx + w * .5f;
    const float z0 = cz - d * .5f, z1 = cz + d * .5f;
    const float y1 = y0 + h;
    const float sr = r * .72f, sg = g * .72f, sb = b * .72f;
    const float tr = std::min(1.f, r * 1.18f), tg = std::min(1.f, g * 1.18f), tb = std::min(1.f, b * 1.18f);
    // side faces
    pushVertex(v, x0, y0, z0, sr, sg, sb); pushVertex(v, x0, y0, z1, sr, sg, sb); pushVertex(v, x0, y1, z1, r, g, b);
    pushVertex(v, x0, y0, z0, sr, sg, sb); pushVertex(v, x0, y1, z1, r, g, b); pushVertex(v, x0, y1, z0, r, g, b);
    pushVertex(v, x1, y0, z0, sr, sg, sb); pushVertex(v, x1, y1, z1, r, g, b); pushVertex(v, x1, y0, z1, sr, sg, sb);
    pushVertex(v, x1, y0, z0, sr, sg, sb); pushVertex(v, x1, y1, z0, r, g, b); pushVertex(v, x1, y1, z1, r, g, b);
    // rear/front
    pushVertex(v, x0, y0, z0, sr, sg, sb); pushVertex(v, x1, y0, z0, sr, sg, sb); pushVertex(v, x1, y1, z0, r, g, b);
    pushVertex(v, x0, y0, z0, sr, sg, sb); pushVertex(v, x1, y1, z0, r, g, b); pushVertex(v, x0, y1, z0, r, g, b);
    pushVertex(v, x0, y0, z1, sr, sg, sb); pushVertex(v, x1, y1, z1, r, g, b); pushVertex(v, x1, y0, z1, sr, sg, sb);
    pushVertex(v, x0, y0, z1, sr, sg, sb); pushVertex(v, x0, y1, z1, r, g, b); pushVertex(v, x1, y1, z1, r, g, b);
    // top
    pushVertex(v, x0, y1, z0, tr, tg, tb); pushVertex(v, x1, y1, z0, tr, tg, tb); pushVertex(v, x1, y1, z1, tr, tg, tb);
    pushVertex(v, x0, y1, z0, tr, tg, tb); pushVertex(v, x1, y1, z1, tr, tg, tb); pushVertex(v, x0, y1, z1, tr, tg, tb);
    if (darkBottom) {
        pushVertex(v, x0, y0, z0, 0.05f, 0.06f, 0.07f); pushVertex(v, x0, y0, z1, 0.05f, 0.06f, 0.07f);
        pushVertex(v, x1, y0, z1, 0.05f, 0.06f, 0.07f);
        pushVertex(v, x0, y0, z0, 0.05f, 0.06f, 0.07f); pushVertex(v, x1, y0, z1, 0.05f, 0.06f, 0.07f);
        pushVertex(v, x1, y0, z0, 0.05f, 0.06f, 0.07f);
    }
}

struct ModelFace {
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
              float scale, float scaleZ, bool ego) {
    quad(v, x - 1.0f, z - 2.2f, x + 1.0f, z + 2.2f, .012f, .022f, .027f, .032f);
    const float cx = 1.428f, cz = .029f;
    const int step = ego ? 2 : 4;
    for (int i = 0; i < faces.size(); i += step) {
        const ModelFace &f = faces[i];
        QVector3D col = materialColor(f.material, ego);
        const float shade = .62f + .38f * std::max(0.0f, f.n.y());
        col *= shade;
        const float xs[3] = {f.a.x(), f.b.x(), f.c.x()};
        const float ys[3] = {f.a.y(), f.b.y(), f.c.y()};
        const float zs[3] = {f.a.z(), f.b.z(), f.c.z()};
        for (int k = 0; k < 3; ++k) {
            const float wx = x + (zs[k] - cz) * scaleZ;
            const float wz = z + (xs[k] - cx) * scale;
            pushVertex(v, wx, std::max(0.015f, ys[k] * scale), wz, col.x(), col.y(), col.z());
        }
    }
}

static QVector<ModelFace> *g_modelFaces = nullptr;

void tri3(std::vector<float> &v, const QVector3D &a, const QVector3D &b,
          const QVector3D &c, float r, float g, float bl) {
    pushVertex(v, a.x(), a.y(), a.z(), r, g, bl);
    pushVertex(v, b.x(), b.y(), b.z(), r, g, bl);
    pushVertex(v, c.x(), c.y(), c.z(), r, g, bl);
}

void quad3(std::vector<float> &v, const QVector3D &a, const QVector3D &b,
           const QVector3D &c, const QVector3D &d, float r, float g, float bl) {
    tri3(v, a, b, c, r, g, bl);
    tri3(v, a, c, d, r, g, bl);
}

void carMesh(std::vector<float> &v, float x, float z, const QString &cls, bool ego) {
    const bool truck = cls == QStringLiteral("truck");
    const bool bus = cls == QStringLiteral("bus");
    const bool person = cls == QStringLiteral("person");
    const bool twoWheel = cls == QStringLiteral("bicycle") || cls == QStringLiteral("motorcycle");
    const float w = person ? .55f : (twoWheel ? .8f : (truck ? 2.25f : (bus ? 2.45f : 1.9f)));
    const float h = person ? 1.65f : (twoWheel ? 1.35f : (truck ? 2.7f : (bus ? 2.8f : 1.45f)));
    const float d = person ? .5f : (twoWheel ? 1.7f : (truck ? 7.2f : (bus ? 9.5f : 4.4f)));
    const float r = ego ? .20f : (truck ? .24f : (bus ? .20f : .36f));
    const float g = ego ? .28f : (truck ? .28f : (bus ? .30f : .41f));
    const float b = ego ? .38f : (truck ? .31f : (bus ? .35f : .46f));

    // soft ground shadow
    quad(v, x - w * .62f, z - d * .55f, x + w * .62f, z + d * .55f, .012f, .025f, .03f, .035f);

    if (person) {
        box3(v, x, .02f, z, .5f, .9f, .4f, .30f, .38f, .42f);
        box3(v, x, .92f, z, .34f, .55f, .30f, .38f, .46f, .50f);
        return;
    }
    if (twoWheel) {
        box3(v, x, .02f, z, .34f, .62f, 1.35f, .22f, .28f, .32f);
        box3(v, x, .58f, z, .46f, .58f, .42f, .38f, .46f, .50f);
        box3(v, x, 1.10f, z, .32f, .26f, .32f, .52f, .56f, .58f);
        return;
    }
    if (truck) {
        box3(v, x, .18f, z - d * .28f, w * .94f, h * .68f, d * .26f, r, g, b);
        box3(v, x, .18f, z + d * .18f, w, h * .92f, d * .64f, r * .9f, g * .9f, b * .9f);
    } else if (bus) {
        box3(v, x, .18f, z, w, h * .90f, d, r, g, b);
        for (int i = -2; i <= 2; ++i)
            box3(v, x - w * .505f, h * .48f, z + i * d * .16f, .03f, h * .30f, d * .11f, .06f, .10f, .13f);
    } else {
        const float y0 = h * .58f, y1 = h * 1.28f;
        const float lw = w * .82f, uw = w * .56f;
        const float ld = d * .58f, ud = d * .40f;
        const float zl = z - ld * .5f, zrf = z + ld * .5f;
        const float zu = z - ud * .18f, zuf = z + ud * .5f;
        // lower body and tapered cabin/roof
        box3(v, x, .18f, z, w, h * .48f, d, r, g, b);
        QVector3D bl(x - lw * .5f, y0, zl), br(x + lw * .5f, y0, zl);
        QVector3D fl(x - lw * .5f, y0, zrf), fr(x + lw * .5f, y0, zrf);
        QVector3D ubl(x - uw * .5f, y1, zu), ubr(x + uw * .5f, y1, zu);
        QVector3D ufl(x - uw * .5f, y1, zuf), ufr(x + uw * .5f, y1, zuf);
        quad3(v, bl, br, ubr, ubl, r * .85f, g * .85f, b * .85f);   // rear slope
        quad3(v, fl, fr, ufr, ufl, r * .95f, g * .95f, b * .95f);   // front slope
        quad3(v, bl, ubl, ufl, fl, r, g, b);                        // left side
        quad3(v, br, ubr, ufr, fr, r, g, b);                        // right side
        quad3(v, ubl, ubr, ufr, ufl, r * 1.10f, g * 1.10f, b * 1.10f); // roof
        // rear/side glass, slightly inset
        const float gy0 = y0 + h * .10f, gy1 = y1 - h * .06f;
        tri3(v, QVector3D(x - lw * .34f, gy0, zl + .035f),
             QVector3D(x + lw * .34f, gy0, zl + .035f),
             QVector3D(x + uw * .32f, gy1, zu + .02f), .025f, .055f, .075f);
        tri3(v, QVector3D(x - lw * .34f, gy0, zl + .035f),
             QVector3D(x + uw * .32f, gy1, zu + .02f),
             QVector3D(x - uw * .32f, gy1, zu + .02f), .025f, .055f, .075f);
        box3(v, x - lw * .505f, y0 + .12f, z - d * .09f, .022f, h * .26f, d * .24f, .04f, .08f, .10f);
        box3(v, x + lw * .505f, y0 + .12f, z - d * .09f, .022f, h * .26f, d * .24f, .04f, .08f, .10f);
        // hood/trunk planes give the body a car silhouette
        quad3(v, QVector3D(x - w * .48f, h * .66f, z + d * .18f),
              QVector3D(x + w * .48f, h * .66f, z + d * .18f),
              QVector3D(x + uw * .5f, y1, zuf),
              QVector3D(x - uw * .5f, y1, zuf), r * .92f, g * .92f, b * .92f);
    }
    // wheels
    for (int sx = -1; sx <= 1; sx += 2) {
        for (int sz = -1; sz <= 1; sz += 2) {
            box3(v, x + sx * (w * .51f), .02f, z + sz * (d * .30f),
                 .18f, h * .22f, .55f, .035f, .04f, .045f);
        }
    }
    if (!person && !truck && !bus) {
        // Rear glass, trunk lip and bumper make the chase view read as a car.
        box3(v, x, h * .72f, z - d * .505f, w * .58f, h * .25f, .035f, .035f, .065f, .085f);
        box3(v, x, h * .52f, z - d * .50f, w * .94f, .09f, .11f, r * .75f, g * .75f, b * .75f);
        box3(v, x, h * .20f, z - d * .50f, w * .98f, .11f, .10f, .08f, .09f, .10f);
        box3(v, x, h * .88f, z - d * .04f, w * .80f, .035f, d * .18f, r * .7f, g * .7f, b * .7f);
    }
    // lamps: neutral amber/white, never hazard red
    box3(v, x - w * .31f, h * .40f, z - d * .51f, .24f, .10f, .03f, .92f, .82f, .56f);
    box3(v, x + w * .31f, h * .40f, z - d * .51f, .24f, .10f, .03f, .92f, .82f, .56f);
    if (ego) {
        box3(v, x, h * .84f, z + d * .02f, w * .55f, .035f, d * .21f, .38f, .68f, .78f);
    }
}


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
        const QStringList parts = line.split(' ', QString::SkipEmptyParts);
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
} // namespace

TeslaSceneItem::TeslaSceneItem(QQuickItem *parent) : QQuickFramebufferObject(parent) {
    setFlag(ItemHasContents, true);
    auto *timer = new QTimer(this);
    timer->setInterval(16);
    connect(timer, &QTimer::timeout, this, [this]() { update(); });
    timer->start();
}

QVariantList TeslaSceneItem::boxes() const {
    QMutexLocker locker(&mutex_);
    return boxes_;
}
void TeslaSceneItem::setBoxes(const QVariantList &boxes) {
    {
        QMutexLocker locker(&mutex_);
        boxes_ = boxes;
    }
    // Auditable bridge from real tracker output to scene objects. This file
    // contains no demo objects and no metric distance claim.
    QJsonArray objects;
    qint64 newest_detect_ns = sourceTimestampNs();
    for (const QVariant &entry : boxes) {
        const QVariantMap b = entry.toMap();
        const QString cls = b.value(QStringLiteral("className")).toString();
        const bool predicted = b.value(QStringLiteral("predicted")).toBool();
        const bool mappable = cls == QStringLiteral("car") || cls == QStringLiteral("truck") ||
                              cls == QStringLiteral("bus") || cls == QStringLiteral("person") ||
                              cls == QStringLiteral("motorcycle") || cls == QStringLiteral("bicycle");
        QJsonObject o;
        o.insert(QStringLiteral("track_id"), static_cast<qint64>(b.value(QStringLiteral("trackId")).toULongLong()));
        o.insert(QStringLiteral("class"), cls);
        o.insert(QStringLiteral("confidence"), b.value(QStringLiteral("confidence")).toDouble());
        o.insert(QStringLiteral("bbox"), QJsonArray{b.value(QStringLiteral("x")).toDouble(),
                                                     b.value(QStringLiteral("y")).toDouble(),
                                                     b.value(QStringLiteral("width")).toDouble(),
                                                     b.value(QStringLiteral("height")).toDouble()});
        o.insert(QStringLiteral("predicted"), predicted);
        o.insert(QStringLiteral("missed"), b.value(QStringLiteral("missed")).toInt());
        o.insert(QStringLiteral("track_state"), predicted ? QStringLiteral("PREDICTED")
                                                          : QStringLiteral("CONFIRMED"));
        o.insert(QStringLiteral("spatial_validity"),
                 (!predicted && mappable) ? QStringLiteral("RELATIVE_ONLY") : QStringLiteral("UNKNOWN"));
        const qint64 detect_ns = static_cast<qint64>(b.value(QStringLiteral("detectTimestampNs")).toULongLong());
        newest_detect_ns = std::max(newest_detect_ns, detect_ns);
        o.insert(QStringLiteral("detect_timestamp_ns"), detect_ns);
        objects.append(o);
    }
    QJsonObject root;
    root.insert(QStringLiteral("source_id"), sourceKey());
    root.insert(QStringLiteral("source_frame_id"), static_cast<qint64>(sourceFrameId()));
    root.insert(QStringLiteral("source_timestamp_ns"), newest_detect_ns);
    root.insert(QStringLiteral("objects"), objects);
    const QByteArray payload = QJsonDocument(root).toJson(QJsonDocument::Compact);
    const QString tmp = QStringLiteral("/dev/shm/tesla_live_objects.json.tmp");
    const QString out = QStringLiteral("/dev/shm/tesla_live_objects.json");
    QFile f(tmp);
    if (f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        f.write(payload);
        f.close();
        QFile::remove(out);
        QFile::rename(tmp, out);
    }
    emit boxesChanged();
    update();
}
QVariantMap TeslaSceneItem::laneState() const {
    QMutexLocker locker(&mutex_);
    return lane_state_;
}
void TeslaSceneItem::setLaneState(const QVariantMap &state) {
    {
        QMutexLocker locker(&mutex_);
        if (lane_state_ == state) return;
        lane_state_ = state;
    }
    emit laneStateChanged();
    update();
}
QString TeslaSceneItem::displayView() const {
    QMutexLocker locker(&mutex_);
    return display_view_;
}
void TeslaSceneItem::setDisplayView(const QString &view) {
    {
        QMutexLocker locker(&mutex_);
        if (display_view_ == view) return;
        display_view_ = view;
    }
    emit displayViewChanged();
    update();
}
void TeslaSceneItem::setRoadQuadrant(int quadrant) {
    quadrant = std::clamp(quadrant, 0, 3);
    if (road_quadrant_.exchange(quadrant) == quadrant) return;
    emit roadQuadrantChanged();
    update();
}
void TeslaSceneItem::setOnline(bool online) {
    if (online_.exchange(online) == online) return;
    emit onlineChanged();
    update();
}
void TeslaSceneItem::setDemoMode(bool demo) {
    if (demo_mode_.exchange(demo) == demo) return;
    emit demoModeChanged();
    update();
}
QString TeslaSceneItem::sourceKey() const {
    QMutexLocker locker(&mutex_);
    return source_key_;
}
void TeslaSceneItem::setSourceKey(const QString &key) {
    {
        QMutexLocker locker(&mutex_);
        if (source_key_ == key) return;
        source_key_ = key;
    }
    emit sourceKeyChanged();
    update();
}

void TeslaSceneItem::setSourceFrameId(qulonglong id) {
    if (source_frame_id_.exchange(id) == id) return;
    emit sourceFrameIdChanged();
}
void TeslaSceneItem::setSourceTimestampNs(qint64 ns) {
    if (source_timestamp_ns_.exchange(ns) == ns) return;
    emit sourceTimestampNsChanged();
}

class SceneRenderer : public QQuickFramebufferObject::Renderer {
public:
    SceneRenderer() { clock_.start(); }

    QOpenGLFramebufferObject *createFramebufferObject(const QSize &size) override {
        QOpenGLFramebufferObjectFormat fmt;
        fmt.setAttachment(QOpenGLFramebufferObject::Depth);
        fmt.setSamples(0);
        return new QOpenGLFramebufferObject(size, fmt);
    }

    void synchronize(QQuickFramebufferObject *item) override {
        auto *scene = static_cast<TeslaSceneItem *>(item);
        const QVariantList list = scene->boxes();
        QVector<SourceBox> next;
        next.reserve(list.size());
        for (const QVariant &entry : list) {
            const QVariantMap map = entry.toMap();
            SourceBox b;
            b.id = map.value(QStringLiteral("trackId")).toInt();
            b.cls = map.value(QStringLiteral("className")).toString();
            b.x = map.value(QStringLiteral("x")).toDouble();
            b.y = map.value(QStringLiteral("y")).toDouble();
            b.w = map.value(QStringLiteral("width")).toDouble();
            b.h = map.value(QStringLiteral("height")).toDouble();
            b.confidence = map.value(QStringLiteral("confidence")).toDouble();
            b.missed = map.value(QStringLiteral("missed")).toInt();
            b.predicted = map.value(QStringLiteral("predicted")).toBool();
            // Stale/predicted tracks are retained by the tracker for continuity,
            // but must not be drawn as if they were current detections.
            if (b.id > 0 && b.w > 2 && b.h > 2 && !b.predicted && b.missed == 0 &&
                b.confidence >= 0.20)
                next.push_back(b);
        }
        boxes_ = next;
        lane_state_ = scene->laneState();
        display_view_ = scene->displayView();
        road_quadrant_ = scene->roadQuadrant();
        online_ = scene->online();
        demo_ = scene->demoMode();
        viewport_ = scene->size().toSize();
        const QString key = scene->sourceKey();
        if (key != source_key_) {
            source_key_ = key;
            live_.clear();
        }
        const double now = clock_.nsecsElapsed() / 1e9;
        dt_ = std::clamp(now - last_time_, 0.001, 0.12);
        last_time_ = now;
        elapsed_ = now;
    }

    void render() override {
        auto *ctx = QOpenGLContext::currentContext();
        if (!ctx) return;
        auto *f = ctx->functions();
        auto checkGl = [f](const char *tag) {
            const GLenum e = f->glGetError();
            if (e != GL_NO_ERROR) qWarning() << "Tesla GL error" << tag << e;
        };
        checkGl("render_start");
        if (!program_) initGL();
        if (!program_) return;
        checkGl("initGL");

        f->glViewport(0, 0, static_cast<GLint>(viewport_.width()), static_cast<GLint>(viewport_.height()));
        checkGl("viewport");
        f->glDisable(GL_BLEND);
        f->glEnable(GL_DEPTH_TEST);
        f->glDepthFunc(GL_LEQUAL);
        f->glClearColor(0.018f, 0.028f, 0.036f, 1.0f);
        f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        checkGl("clear");

        std::vector<float> vertices;
        vertices.reserve(24000);
        buildScene(vertices);
        checkGl("buildScene");
        if (vertices.empty()) return;

        vao_.bind();
        vbo_.bind();
        vbo_.allocate(vertices.data(), static_cast<int>(vertices.size() * sizeof(float)));
        checkGl("vbo_allocate");
        program_->bind();
        const int pos = program_->attributeLocation("aPos");
        const int col = program_->attributeLocation("aColor");
        program_->enableAttributeArray(pos);
        program_->enableAttributeArray(col);
        program_->setAttributeBuffer(pos, GL_FLOAT, 0, 3, 6 * sizeof(float));
        program_->setAttributeBuffer(col, GL_FLOAT, 3 * sizeof(float), 3, 6 * sizeof(float));
        QMatrix4x4 projection;
        const float aspect = viewport_.height() > 0 ? float(viewport_.width()) / float(viewport_.height()) : 1.7f;
        const bool compact = viewport_.height() < 160;
        projection.perspective(compact ? 44.0f : 52.0f, aspect, 0.08f, 180.0f);
        // Chase camera looks from -Z toward +Z; flip X so screen-right is
        // vehicle-right and left/right LVT1 lane styles are not mirrored.
        projection.scale(-1.0f, -1.0f, 1.0f);
        QMatrix4x4 view;
        if (compact)
            view.lookAt(QVector3D(-1.75f, 5.2f, -8.0f), QVector3D(-1.75f, 0.1f, 12.0f), QVector3D(0, 1, 0));
        else
            view.lookAt(QVector3D(-1.75f, 8.2f, -13.5f), QVector3D(-1.75f, 0.2f, 18.0f), QVector3D(0, 1, 0));
        QMatrix4x4 mvp = projection * view;
        program_->setUniformValue("uMVP", mvp);
        checkGl("uniforms");
        f->glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size() / 6));
        checkGl("draw");
        program_->disableAttributeArray(pos);
        program_->disableAttributeArray(col);
        vbo_.release();
        vao_.release();
        drawBillboards(mvp);

        {
            const GLenum sceneError = f->glGetError();
            if (sceneError != GL_NO_ERROR)
                qWarning() << "GL error at TeslaSceneItem::render end" << sceneError;
        }
        frames_++;
        if (elapsed_ - last_report_ > 1.0) {
            // Rendering thread only updates an atomic; no QObject signal here.
            last_report_ = elapsed_;
            frames_ = 0;
        }
    }

private:
    void initGL() {
        program_ = std::make_unique<QOpenGLShaderProgram>();
        const char *vs = R"(#version 300 es
        in vec3 aPos; in vec3 aColor; uniform mat4 uMVP; out vec3 vColor;
        void main(){ vColor=aColor; gl_Position=uMVP*vec4(aPos,1.0); })";
        const char *fs = R"(#version 300 es
        precision mediump float; in vec3 vColor; out vec4 fragColor;
        void main(){ fragColor=vec4(vColor,1.0); })";
        if (!program_->addShaderFromSourceCode(QOpenGLShader::Vertex, vs) ||
            !program_->addShaderFromSourceCode(QOpenGLShader::Fragment, fs) ||
            !program_->link()) {
            qWarning("TeslaScene shader failed: %s", qPrintable(program_->log()));
            program_.reset();
            return;
        }
        vbo_.create();
        vao_.create();
        vbo_.setUsagePattern(QOpenGLBuffer::DynamicDraw);

        // The billboard traffic path is retired: real targets are now 3D
        // meshes. Do not create its texture/VAO/program.
    }

    void addBillboard(float x, float z, const QString &cls) {
        Billboard b;
        b.x = x; b.z = z;
        if (cls == "truck") { b.w = 2.5f; b.tint = QVector3D(.42f, .46f, .50f); }
        else if (cls == "bus") { b.w = 2.7f; b.tint = QVector3D(.34f, .44f, .52f); }
        else if (cls == "person") { b.w = .75f; b.tint = QVector3D(.65f, .66f, .58f); }
        else { b.w = 1.95f; b.tint = QVector3D(.58f, .62f, .67f); }
        b.h = b.w * 740.0f / 680.0f;
        billboards_.push_back(b);
    }

    void addDemo(std::vector<float> &v) {
        // The original demo's motion is deliberately separate from LIVE.
        const float phase = float(std::fmod(elapsed_ * 0.055, 1.0));
        const char *classes[] = {"car", "car", "car", "car", "car", "car"};
        const float lanes[] = {-2.4f, 2.4f, 0.0f, 3.1f, -3.1f, 1.2f};
        for (int i = 0; i < 5; ++i) {
            const float z = 8.0f + std::fmod(10.0f + i * 11.0f + phase * 58.0f, 58.0f);
            addBillboard(lanes[i], z, QString::fromLatin1(classes[i]));
        }
        // dash motion makes the DEMO world visibly move without claiming metric speed.
        for (int i = 0; i < 18; ++i) {
            const float z = 5.0f + std::fmod(i * 4.0f + elapsed_ * 8.0f, 72.0f);
            quad(v, -0.035f, z, 0.035f, z + 1.6f, .021f, .55f, .60f, .63f);
        }
    }

    void addLive(std::vector<float> &v) {
        if (!online_) {
            live_.clear();
            return;
        }
        const double now = elapsed_;
        // Box coordinates are in the AI display view. In quad mode the AI
        // statuses with display_view=quad; subtract the selected quadrant
        // offset so navigation/cabin quadrants cannot leak into the road BEV.
        double sourceW = 1280.0, sourceH = 720.0, offX = 0.0, offY = 0.0;
        if (display_view_ == QStringLiteral("quad")) {
            sourceW = 640.0; sourceH = 360.0;
            offX = (road_quadrant_ % 2) * 640.0;
            offY = (road_quadrant_ / 2) * 360.0;
        }
        for (const SourceBox &b : boxes_) {
            if (!(b.cls == "car" || b.cls == "truck" || b.cls == "bus" ||
                  b.cls == "person" || b.cls == "motorcycle" || b.cls == "bicycle")) continue;
            const double boxCx = b.x + b.w * .5;
            const double boxBottom = b.y + b.h;
            if (boxCx < offX || boxCx > offX + sourceW ||
                boxBottom < offY || boxBottom > offY + sourceH)
                continue;
            const double cx = (boxCx - offX) / sourceW;
            const double bottom = (boxBottom - offY) / sourceH;
            const float tx = std::clamp(float(-1.75 + (cx - .5) * 14.0), -6.8f, 6.8f);
            // Non-calibrated relative depth: bbox width is a scale cue, not a
            // metric distance. It separates near/far detections much better
            // than a constant bottom-to-depth line and never yields metres.
            const bool narrow = b.cls == "person" || b.cls == "bicycle" || b.cls == "motorcycle";
            const float widthDepth = (narrow ? 1200.0f : 2600.0f) / std::max(24.0f, float(b.w));
            const float bottomDepth = 3.8f + float(1.0 - bottom) * 42.0f;
            const float tz = std::clamp(0.72f * widthDepth + 0.28f * bottomDepth, 4.0f, 62.0f);
            auto it = live_.find(b.id);
            if (it == live_.end()) {
                live_.insert(b.id, LiveCar{tx, tz, b.cls, now});
            } else {
                it->cls = b.cls;
                const float k = std::min(1.0f, float(dt_ * 5.0));
                it->x += (tx - it->x) * k;
                it->z += (tz - it->z) * k;
                it->seen = now;
            }
        }
        for (auto it = live_.begin(); it != live_.end();) {
            // Matches the Qt tracker's 900 ms remove window; no ghost target.
            if (now - it->seen > 0.9) it = live_.erase(it); else ++it;
        }
        for (const LiveCar &car : live_) carMesh(v, car.x, car.z, car.cls, false);
    }

    void drawBillboards(const QMatrix4x4 &mvp) {
        if (!texture_ || billboards_.empty()) return;
        auto *ctx = QOpenGLContext::currentContext();
        if (!ctx) return;
        auto *f = ctx->functions();
        std::sort(billboards_.begin(), billboards_.end(), [](const Billboard &a, const Billboard &b) { return a.z > b.z; });
        std::vector<float> verts;
        verts.reserve(billboards_.size() * 6 * 5);
        for (const Billboard &b : billboards_) {
            const float x0 = b.x - b.w * .5f, x1 = b.x + b.w * .5f;
            const float y0 = .02f, y1 = b.h;
            const float z = b.z;
            auto vtx = [&](float x, float y, float u, float v) {
                verts.insert(verts.end(), {x, y, z, u, v});
            };
            vtx(x0, y0, 0, 1); vtx(x1, y0, 1, 1); vtx(x1, y1, 1, 0);
            vtx(x0, y0, 0, 1); vtx(x1, y1, 1, 0); vtx(x0, y1, 0, 0);
        }
        if (!billboard_program_) return;
        billboard_vao_.bind(); billboard_vbo_.bind();
        billboard_vbo_.allocate(verts.data(), int(verts.size() * sizeof(float)));
        billboard_program_->bind();
        billboard_program_->setUniformValue("uMVP", mvp);
        billboard_program_->setUniformValue("uTex", 0);
        const int p = billboard_program_->attributeLocation("aPos");
        const int uv = billboard_program_->attributeLocation("aUV");
        billboard_program_->enableAttributeArray(p);
        billboard_program_->enableAttributeArray(uv);
        billboard_program_->setAttributeBuffer(p, GL_FLOAT, 0, 3, 5 * sizeof(float));
        billboard_program_->setAttributeBuffer(uv, GL_FLOAT, 3 * sizeof(float), 2, 5 * sizeof(float));
        f->glEnable(GL_BLEND);
        f->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        f->glDepthMask(GL_FALSE);
        texture_->bind(0);
        f->glDrawArrays(GL_TRIANGLES, 0, GLsizei(verts.size() / 5));
        texture_->release();
        f->glDepthMask(GL_TRUE);
        f->glDisable(GL_BLEND);
        billboard_program_->disableAttributeArray(p);
        billboard_program_->disableAttributeArray(uv);
        billboard_vbo_.release();
        billboard_vao_.release();
        billboards_.clear();
    }

    void laneEdge(std::vector<float> &v, float x, const QString &type) {
        // The LVT1 packet carries only nearest boundary type and counts.
        // It never carries lane pixel coordinates. We therefore restyle the
        // existing schematic road edge and never invent extra lane geometry.
        if (type == QStringLiteral("solid")) {
            quad(v, x - .045f, 0, x + .045f, 90, .031f, .65f, .73f, .76f);
            return;
        }
        if (type == QStringLiteral("dashed")) {
            for (int i = 0; i < 22; ++i) {
                const float z0 = i * 4.0f + .7f;
                quad(v, x - .045f, z0, x + .045f, z0 + 2.35f, .031f, .48f, .55f, .58f);
            }
            return;
        }
        // Unknown: faint dotted guide, deliberately not a solid/dashed claim.
        for (int i = 0; i < 30; ++i) {
            const float z0 = i * 3.0f + .5f;
            quad(v, x - .025f, z0, x + .025f, z0 + .8f, .026f, .18f, .24f, .26f);
        }
    }

    void buildScene(std::vector<float> &v) {
        billboards_.clear();
        // Four same-direction lanes, 3.5 m each. The FSD blue path is only
        // the ego lane (second from the left), not the whole road.
        const float roadHalf = 7.0f;
        const float egoX = -1.75f;
        quad(v, -90, -12, 90, 120, 0.0f, .030f, .046f, .050f);
        quad(v, -80, -8, -7.8f, 112, .006f, .045f, .070f, .062f);
        quad(v, 7.8f, -8, 80, 112, .006f, .045f, .070f, .062f);
        // Dark distant sky wall; buildings are only schematic background.
        box3(v, 0, 0, 116, 170, 34, 2, .020f, .032f, .044f, false);
        quad(v, -7.5f, 0, 7.5f, 92, .018f, .105f, .125f, .135f);
        quad(v, -7.25f, 0, 7.25f, 92, .026f, .105f, .115f, .120f);
        // Ego-lane FSD path.
        quad(v, egoX - 1.55f, 0, egoX + 1.55f, 90, .028f, .060f, .25f, .40f);
        // Road edges follow the LVT1 nearest boundary type. Counts are
        // displayed in the lane card, not converted into extra lanes.
        const bool laneValid = lane_state_.value(QStringLiteral("valid")).toBool();
        const QString leftType = laneValid
            ? lane_state_.value(QStringLiteral("left_type")).toString()
            : QStringLiteral("unknown");
        const QString rightType = laneValid
            ? lane_state_.value(QStringLiteral("right_type")).toString()
            : QStringLiteral("unknown");
        laneEdge(v, -7.105f, leftType);
        laneEdge(v, 7.105f, rightType);
        // Dashed lane boundaries at -3.5, 0, +3.5.
        for (float laneX : {-3.5f, 0.0f, 3.5f}) {
            for (int i = 0; i < 22; ++i) {
                const float z0 = i * 4.0f + .7f;
                quad(v, laneX - .045f, z0, laneX + .045f, z0 + 2.35f, .032f, .48f, .55f, .58f);
            }
        }
        // Roadside grid and city silhouettes.
        for (int i = 0; i < 8; ++i) {
            const float z = 8.0f + i * 9.0f;
            quad(v, -30, z, -7.6f, z + .06f, .022f, .085f, .11f, .13f);
            quad(v, 7.6f, z, 30, z + .06f, .022f, .085f, .11f, .13f);
        }
        for (int side = -1; side <= 1; side += 2) {
            for (int i = 0; i < 8; ++i) {
                const float x = side * (9.0f + i * 3.0f);
                quad(v, x, 0, x + .05f, 92, .025f, .065f, .085f, .10f);
            }
        }
        for (int i = 0; i < 13; ++i) {
            const float x = -30.0f + i * 5.0f;
            const float h = 3.0f + float((i * 7) % 9);
            box3(v, x, 0.0f, 86.0f + (i % 3) * 2.5f, 3.4f, h, 2.0f,
                 .045f, .065f, .075f, false);
        }
        for (int side = -1; side <= 1; side += 2) {
            for (int i = 0; i < 4; ++i) {
                box3(v, side * (14.0f + i * 3.5f), 0.0f, 26.0f + i * 13.0f,
                     4.0f + i * 1.2f, 4.0f + (i % 3) * 3.0f, 4.0f,
                     .035f, .052f, .062f, false);
            }
        }
        if (demo_) addDemo(v); else addLive(v);
        // Ego is a real 3D member of the same projected world, not a QML
        // overlay. Its shape is a presentation model, not a measurement.
        carMesh(v, -1.75f, 0.65f, QStringLiteral("car"), true);
    }

    std::unique_ptr<QOpenGLShaderProgram> program_;
    std::unique_ptr<QOpenGLShaderProgram> billboard_program_;
    QOpenGLTexture *texture_ = nullptr;
    std::vector<Billboard> billboards_;
    QOpenGLBuffer billboard_vbo_{QOpenGLBuffer::VertexBuffer};
    QOpenGLVertexArrayObject billboard_vao_;
    QOpenGLBuffer vbo_{QOpenGLBuffer::VertexBuffer};
    QOpenGLVertexArrayObject vao_;
    QElapsedTimer clock_;
    double last_time_ = 0.0;
    double elapsed_ = 0.0;
    float dt_ = .016f;
    double last_report_ = 0.0;
    int frames_ = 0;
    QVector<SourceBox> boxes_;
    QVariantMap lane_state_;
    QString display_view_ = QStringLiteral("front");
    int road_quadrant_ = 1;
    QHash<int, LiveCar> live_;
    QString source_key_;
    bool online_ = false;
    bool demo_ = false;
    QSize viewport_;
};


QQuickFramebufferObject::Renderer *TeslaSceneItem::createRenderer() const {
    return new SceneRenderer();
}
