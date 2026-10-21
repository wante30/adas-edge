#include "video_texture_item.h"
#include "video_item.h"

#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QSGSimpleTextureNode>
#include <QSGTexture>
#include <QQuickWindow>
#include <QImage>
#include <chrono>

class NativeTexture : public QSGTexture {
public:
    NativeTexture(unsigned int id, const QSize &size) : id_(id), size_(size) {}
    ~NativeTexture() override = default;

    int textureId() const override { return static_cast<int>(id_); }
    QSize textureSize() const override { return size_; }
    bool hasAlphaChannel() const override { return false; }
    bool hasMipmaps() const override { return false; }
    void bind() override {
        if (auto *context = QOpenGLContext::currentContext()) {
            context->functions()->glBindTexture(GL_TEXTURE_2D, id_);
            updateBindOptions();
        }
    }

private:
    unsigned int id_ = 0;
    QSize size_;
};

VideoTextureItem::VideoTextureItem(QQuickItem *parent) : QQuickItem(parent) {
    setFlag(ItemHasContents, true);
    qWarning() << "VideoTextureItem constructed";
}

VideoTextureItem::~VideoTextureItem() = default;

void VideoTextureItem::setSource(QQuickItem *source) {
    if (sourceItem_ == source) return;
    sourceItem_ = source;
    video_ = qobject_cast<VideoItem *>(source);
    update();
    emit sourceChanged();
}

QSGNode *VideoTextureItem::updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *) {
    qWarning() << "VideoTextureItem updatePaintNode size" << width() << height() << "old" << (oldNode != nullptr);
    if (!video_) return oldNode;

    auto *node = static_cast<QSGSimpleTextureNode *>(oldNode);
    if (!node) {
        node = new QSGSimpleTextureNode();
        node->setOwnsTexture(false);
        textureNode_ = node;
    }

    if (qEnvironmentVariableIsSet("SG_TEST_IMAGE")) {
        if (!testImageLoaded_ && window()) {
            QImage image(64, 64, QImage::Format_RGBA8888);
            image.fill(Qt::red);
            for (int y = 0; y < 32; ++y)
                for (int x = 0; x < 32; ++x) image.setPixelColor(x, y, Qt::green);
            for (int y = 0; y < 32; ++y)
                for (int x = 32; x < 64; ++x) image.setPixelColor(x, y, Qt::blue);
            for (int y = 32; y < 64; ++y)
                for (int x = 0; x < 32; ++x) image.setPixelColor(x, y, Qt::yellow);
            qWarning() << "SG_TEST_IMAGE creating" << width() << height();
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
    if (queue) frame = queue->take();

    if (frame && !frame->pixels.empty()) {
        auto *context = QOpenGLContext::currentContext();
        if (!context) return node;
        auto *f = context->functions();
        const QSize frameSize(1280, 720);
        if (!textureId_) {
            f->glGenTextures(1, &textureId_);
            if (!textureId_) return node;
            f->glBindTexture(GL_TEXTURE_2D, textureId_);
            f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            f->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB565, frameSize.width(), frameSize.height(),
                            0, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, frame->pixels.data());
            textureSize_ = frameSize;
        } else {
            f->glBindTexture(GL_TEXTURE_2D, textureId_);
            f->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frameSize.width(), frameSize.height(),
                               GL_RGB, GL_UNSIGNED_SHORT_5_6_5, frame->pixels.data());
        }
        const GLenum error = f->glGetError();
        if (error != GL_NO_ERROR) {
            qWarning() << "VideoTextureItem RGB565 upload failed" << error;
            return node;
        }

        if (!texture_) {
            texture_ = new NativeTexture(textureId_, textureSize_);
            texture_->setFiltering(QSGTexture::Linear);
            texture_->setHorizontalWrapMode(QSGTexture::ClampToEdge);
            texture_->setVerticalWrapMode(QSGTexture::ClampToEdge);
        }
        node->setTexture(texture_);
        node->setOwnsTexture(false);
        node->markDirty(QSGNode::DirtyMaterial | QSGNode::DirtyGeometry);

        const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        const qint64 sourceNs = frame->sourceTimestampNs > 0 ? frame->sourceTimestampNs : frame->producedNs;
        video_->recordExternalUpload(*frame, 0.0,
                                     std::max<qint64>(0, now - sourceNs) / 1.0e6);
    }

    if (texture_) {
        node->setRect(0, 0, width(), height());
        // Temporary full-frame source rect for texture-path validation.
        node->setSourceRect(QRectF(0, 0, textureSize_.width(), textureSize_.height()));
        node->setFiltering(QSGTexture::Linear);
    }
    return node;
}
