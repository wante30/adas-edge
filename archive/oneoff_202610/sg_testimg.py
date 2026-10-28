from pathlib import Path
h=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_texture_item.h')
s=h.read_text(encoding='utf-8')
s=s.replace('''    QSGTexture *texture_ = nullptr;''','''    QSGTexture *texture_ = nullptr;
    bool testImageLoaded_ = false;''',1)
h.write_text(s,encoding='utf-8')
c=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_texture_item.cpp')
s=c.read_text(encoding='utf-8')
s=s.replace('#include <QQuickWindow>','#include <QQuickWindow>\n#include <QImage>',1)
needle='''    auto queue = video_->displayQueue();
    std::shared_ptr<VideoItem::Frame> frame;
    if (queue) frame = queue->take();'''
rep='''    if (qEnvironmentVariableIsSet("SG_TEST_IMAGE")) {
        if (!testImageLoaded_ && window()) {
            QImage image(64, 64, QImage::Format_RGBA8888);
            image.fill(Qt::red);
            for (int y = 0; y < 32; ++y)
                for (int x = 0; x < 32; ++x) image.setPixelColor(x, y, Qt::green);
            for (int y = 0; y < 32; ++y)
                for (int x = 32; x < 64; ++x) image.setPixelColor(x, y, Qt::blue);
            for (int y = 32; y < 64; ++y)
                for (int x = 0; x < 32; ++x) image.setPixelColor(x, y, Qt::yellow);
            if (QSGTexture *test = window()->createTextureFromImage(image)) {
                node->setTexture(test);
                node->setOwnsTexture(true);
                node->setRect(0, 0, width(), height());
                node->setSourceRect(QRectF(0, 0, 1, 1));
                node->markDirty(QSGNode::DirtyMaterial | QSGNode::DirtyGeometry);
                testImageLoaded_ = true;
            } else {
                qWarning() << "SG_TEST_IMAGE createTextureFromImage failed";
            }
        }
        return node;
    }

    auto queue = video_->displayQueue();
    std::shared_ptr<VideoItem::Frame> frame;
    if (queue) frame = queue->take();'''
if needle not in s: raise SystemExit('queue marker')
s=s.replace(needle,rep,1)
c.write_text(s,encoding='utf-8')
print('test image branch')
