from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('#include <QDateTime>\n','#include <QDateTime>\n#include <QFile>\n#include <QJsonArray>\n#include <QJsonDocument>\n#include <QJsonObject>\n')
# insert setters after sourceKey setter
needle='''void TeslaSceneItem::setSourceKey(const QString &key) {
    {
        QMutexLocker locker(&mutex_);
        if (source_key_ == key) return;
        source_key_ = key;
    }
    emit sourceKeyChanged();
    update();
}
'''
rep=needle+'''
void TeslaSceneItem::setSourceFrameId(qulonglong id) {
    if (source_frame_id_.exchange(id) == id) return;
    emit sourceFrameIdChanged();
}
void TeslaSceneItem::setSourceTimestampNs(qint64 ns) {
    if (source_timestamp_ns_.exchange(ns) == ns) return;
    emit sourceTimestampNsChanged();
}
'''
if needle not in s: raise SystemExit('setter marker')
s=s.replace(needle,rep,1)
# write mapping in setBoxes before unlock? replace setBoxes body
old='''void TeslaSceneItem::setBoxes(const QVariantList &boxes) {
    {
        QMutexLocker locker(&mutex_);
        boxes_ = boxes;
    }
    emit boxesChanged();
    update();
}'''
new='''void TeslaSceneItem::setBoxes(const QVariantList &boxes) {
    {
        QMutexLocker locker(&mutex_);
        boxes_ = boxes;
    }
    // Auditable bridge from real tracker output to scene objects. This file
    // contains no demo objects and no metric distance claim.
    QJsonArray objects;
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
        o.insert(QStringLiteral("detect_timestamp_ns"),
                 static_cast<qint64>(b.value(QStringLiteral("detectTimestampNs")).toULongLong()));
        objects.append(o);
    }
    QJsonObject root;
    root.insert(QStringLiteral("source_id"), sourceKey());
    root.insert(QStringLiteral("source_frame_id"), static_cast<qint64>(sourceFrameId()));
    root.insert(QStringLiteral("source_timestamp_ns"), static_cast<qint64>(sourceTimestampNs()));
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
}'''
if old not in s: raise SystemExit('setBoxes marker')
s=s.replace(old,new,1)
# add ego in buildScene at end
s=s.replace('''        if (demo_) addDemo(v); else addLive(v);
        // Ego is the QML rear-view asset above the FBO.''','''        if (demo_) addDemo(v); else addLive(v);
        // Ego is a real 3D member of the same projected world, not a QML
        // overlay. Its shape is a presentation model, not a measurement.
        carMesh(v, -1.75f, 0.65f, QStringLiteral("car"), true);''')
p.write_text(s,encoding='utf-8')
print('mapping/ego patched')
