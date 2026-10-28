from pathlib import Path
h=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.h')
s=h.read_text(encoding='utf-8')
s=s.replace('''    Q_PROPERTY(QString sourceKey READ sourceKey WRITE setSourceKey NOTIFY sourceKeyChanged)''','''    Q_PROPERTY(QString sourceKey READ sourceKey WRITE setSourceKey NOTIFY sourceKeyChanged)
    Q_PROPERTY(qulonglong sourceFrameId READ sourceFrameId WRITE setSourceFrameId NOTIFY sourceFrameIdChanged)
    Q_PROPERTY(qint64 sourceTimestampNs READ sourceTimestampNs WRITE setSourceTimestampNs NOTIFY sourceTimestampNsChanged)''')
s=s.replace('''    void setSourceKey(const QString &key);
    double sceneFps() const { return scene_fps_.load(); }''','''    void setSourceKey(const QString &key);
    qulonglong sourceFrameId() const { return source_frame_id_.load(); }
    void setSourceFrameId(qulonglong id);
    qint64 sourceTimestampNs() const { return source_timestamp_ns_.load(); }
    void setSourceTimestampNs(qint64 ns);
    double sceneFps() const { return scene_fps_.load(); }''')
s=s.replace('''    void sourceKeyChanged();
    void sceneFpsChanged();''','''    void sourceKeyChanged();
    void sourceFrameIdChanged();
    void sourceTimestampNsChanged();
    void sceneFpsChanged();''')
s=s.replace('''    std::atomic<bool> demo_mode_{false};
    std::atomic<double> scene_fps_{0.0};''','''    std::atomic<bool> demo_mode_{false};
    std::atomic<double> scene_fps_{0.0};
    std::atomic<qulonglong> source_frame_id_{0};
    std::atomic<qint64> source_timestamp_ns_{0};''')
h.write_text(s,encoding='utf-8')
print('header properties')
