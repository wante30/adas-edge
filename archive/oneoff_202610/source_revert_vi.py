from pathlib import Path
h=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.h')
s=h.read_text(encoding='utf-8')
s=s.replace('''    Q_INVOKABLE void switchPgvSource(const QString &pgvFile, const QString &sourceLabel);\n''','',1)
s=s.replace('''    double producer_fps_ = 30.0;
    QString producer_ai_frame_file_;
    double producer_ai_hz_ = 10.0;
''','',1)
h.write_text(s,encoding='utf-8')
c=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=c.read_text(encoding='utf-8')
s=s.replace('''    stopProducer();
    producer_fps_ = fps;
    producer_ai_frame_file_ = aiFrameFile;
    producer_ai_hz_ = aiHz;
    display_queue_->clear();''','''    stopProducer();
    display_queue_->clear();''',1)
start=s.find('void VideoItem::switchPgvSource')
if start>=0:
    end=s.find('void VideoItem::stopProducer',start)
    s=s[:start]+s[end:]
c.write_text(s,encoding='utf-8')
print('reverted VideoItem switch')
