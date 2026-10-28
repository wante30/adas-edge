from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
# source timestamp fallback
s=s.replace('''    QJsonArray objects;
    for (const QVariant &entry : boxes) {''','''    QJsonArray objects;
    qint64 newest_detect_ns = sourceTimestampNs();
    for (const QVariant &entry : boxes) {''')
s=s.replace('''        o.insert(QStringLiteral("detect_timestamp_ns"),
                 static_cast<qint64>(b.value(QStringLiteral("detectTimestampNs")).toULongLong()));
        objects.append(o);''','''        const qint64 detect_ns = static_cast<qint64>(b.value(QStringLiteral("detectTimestampNs")).toULongLong());
        newest_detect_ns = std::max(newest_detect_ns, detect_ns);
        o.insert(QStringLiteral("detect_timestamp_ns"), detect_ns);
        objects.append(o);''')
s=s.replace('''    root.insert(QStringLiteral("source_timestamp_ns"), static_cast<qint64>(sourceTimestampNs()));''','''    root.insert(QStringLiteral("source_timestamp_ns"), newest_detect_ns);''')
# camera compact
s=s.replace('''        QMatrix4x4 projection;
        const float aspect = viewport_.height() > 0 ? float(viewport_.width()) / float(viewport_.height()) : 1.7f;
        projection.perspective(52.0f, aspect, 0.08f, 180.0f);
        projection.scale(1.0f, -1.0f, 1.0f);
        QMatrix4x4 view;
        view.lookAt(QVector3D(-1.75f, 8.2f, -13.5f), QVector3D(-1.75f, 0.2f, 18.0f), QVector3D(0, 1, 0));''','''        QMatrix4x4 projection;
        const float aspect = viewport_.height() > 0 ? float(viewport_.width()) / float(viewport_.height()) : 1.7f;
        const bool compact = viewport_.height() < 160;
        projection.perspective(compact ? 44.0f : 52.0f, aspect, 0.08f, 180.0f);
        projection.scale(1.0f, -1.0f, 1.0f);
        QMatrix4x4 view;
        if (compact)
            view.lookAt(QVector3D(-1.75f, 5.2f, -8.0f), QVector3D(-1.75f, 0.1f, 12.0f), QVector3D(0, 1, 0));
        else
            view.lookAt(QVector3D(-1.75f, 8.2f, -13.5f), QVector3D(-1.75f, 0.2f, 18.0f), QVector3D(0, 1, 0));''')
p.write_text(s,encoding='utf-8')
print('camera/timestamp patched')
