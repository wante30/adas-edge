from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=p.read_text(encoding='utf-8')
marker='void VideoItem::setAiRoadQuadrant(int quadrant) {'
code=r'''void VideoItem::setSpatialViewActive(bool value) {
    if (spatial_view_active_.exchange(value) == value) return;
    QFile flag(QStringLiteral("/dev/shm/tesla_scene.active"));
    if (flag.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        flag.write(value ? "1\n" : "0\n");
    }
    emit spatialViewActiveChanged();
}

void VideoItem::writeTeslaSceneTracks(const QVariantList &boxes, uint64_t frameId,
                                      qint64 timestampNs) {
    if (!spatial_view_active_.load()) return;
    QJsonObject root;
    root.insert(QStringLiteral("source"), source_mode_);
    root.insert(QStringLiteral("source_frame_id"), static_cast<qulonglong>(frameId));
    root.insert(QStringLiteral("timestamp_ns"), static_cast<qulonglong>(timestampNs));
    root.insert(QStringLiteral("display_width"), kSourceWidth);
    root.insert(QStringLiteral("display_height"), kSourceHeight);
    root.insert(QStringLiteral("simulated"), false);
    QJsonArray items;
    for (const QVariant &entry : boxes) {
        const QVariantMap box = entry.toMap();
        if (box.isEmpty()) continue;
        QJsonObject item;
        item.insert(QStringLiteral("id"), box.value(QStringLiteral("trackId")).toULongLong());
        item.insert(QStringLiteral("class"), box.value(QStringLiteral("className")).toString());
        item.insert(QStringLiteral("x"), box.value(QStringLiteral("x")).toDouble());
        item.insert(QStringLiteral("y"), box.value(QStringLiteral("y")).toDouble());
        item.insert(QStringLiteral("width"), box.value(QStringLiteral("width")).toDouble());
        item.insert(QStringLiteral("height"), box.value(QStringLiteral("height")).toDouble());
        item.insert(QStringLiteral("vx"), box.value(QStringLiteral("vx")).toDouble());
        item.insert(QStringLiteral("vy"), box.value(QStringLiteral("vy")).toDouble());
        item.insert(QStringLiteral("confidence"), box.value(QStringLiteral("confidence")).toDouble());
        item.insert(QStringLiteral("state"), box.value(QStringLiteral("state")).toString());
        item.insert(QStringLiteral("detect_timestamp_ns"),
                    box.value(QStringLiteral("detectTimestampNs")).toULongLong());
        items.append(item);
    }
    root.insert(QStringLiteral("tracks"), items);
    const QByteArray payload = QJsonDocument(root).toJson(QJsonDocument::Compact);
    const QString target = QStringLiteral("/dev/shm/tesla_scene_tracks.json");
    const QString temporary = target + QStringLiteral(".tmp");
    QFile file(temporary);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) return;
    file.write(payload);
    file.close();
    QFile::remove(target);
    QFile::rename(temporary, target);
}

'''
if marker not in s: raise SystemExit('marker')
s=s.replace(marker,code+marker,1)
p.write_text(s,encoding='utf-8')
print('methods inserted')
