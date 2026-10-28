from pathlib import Path
# VideoItem header
h=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.h')
s=h.read_text(encoding='utf-8')
needle='''    void startPgvProducer(double fps, const QString &pgvFile,
                          const QString &aiFrameFile = QString(),
                          double aiHz = 10.0,
                          const QString &sourceLabel = QStringLiteral("pgv-live"));'''
rep='''    void startPgvProducer(double fps, const QString &pgvFile,
                          const QString &aiFrameFile = QString(),
                          double aiHz = 10.0,
                          const QString &sourceLabel = QStringLiteral("pgv-live"));
    Q_INVOKABLE void switchPgvSource(const QString &pgvFile, const QString &sourceLabel);'''
if needle not in s: raise SystemExit('start method')
s=s.replace(needle,rep,1)
s=s.replace('''    std::atomic<bool> producer_running_{false};''','''    std::atomic<bool> producer_running_{false};
    double producer_fps_ = 30.0;
    QString producer_ai_frame_file_;
    double producer_ai_hz_ = 10.0;''',1)
h.write_text(s,encoding='utf-8')
# cpp store + switch
c=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=c.read_text(encoding='utf-8')
s=s.replace('''    stopProducer();
    display_queue_->clear();
    replay_frames_.clear();
    pgv_source_ = true;''','''    stopProducer();
    producer_fps_ = fps;
    producer_ai_frame_file_ = aiFrameFile;
    producer_ai_hz_ = aiHz;
    display_queue_->clear();
    replay_frames_.clear();
    pgv_source_ = true;''',1)
needle='''void VideoItem::stopProducer() {'''
rep='''void VideoItem::switchPgvSource(const QString &pgvFile, const QString &sourceLabel) {
    if (pgvFile.isEmpty()) return;
    startPgvProducer(producer_fps_, pgvFile, producer_ai_frame_file_,
                     producer_ai_hz_, sourceLabel);
}

void VideoItem::stopProducer() {'''
if needle not in s: raise SystemExit('stop method')
s=s.replace(needle,rep,1)
c.write_text(s,encoding='utf-8')
# main args/context
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''    const QString sourceLabel = stringArg(args, "--source-label");''','''    const QString sourceLabel = stringArg(args, "--source-label");
    const QString pciePgvFile = stringArg(args, "--pcie-pgv-file", QStringLiteral("/dev/shm/pango-video/latest.pgv"));
    const QString localPgvFile = stringArg(args, "--local-pgv-file", QStringLiteral("/dev/shm/pgv_local/latest.pgv"));''',1)
s=s.replace('''    view.rootContext()->setContextProperty(QStringLiteral("sceneEnabled"), sceneEnabled);''','''    view.rootContext()->setContextProperty(QStringLiteral("sceneEnabled"), sceneEnabled);
    view.rootContext()->setContextProperty(QStringLiteral("pciePgvFile"), pciePgvFile);
    view.rootContext()->setContextProperty(QStringLiteral("localPgvFile"), localPgvFile);''',1)
p.write_text(s,encoding='utf-8')
print('backend switch')
