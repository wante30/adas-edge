#include "qsg_video_item.h"

#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QSGTexture>
#include <algorithm>
#include <memory>

class QsgVideoItem::Rgb565Texture final : public QSGDynamicTexture {
public:
    explicit Rgb565Texture(VideoItem *source) : source_(source) {}

    ~Rgb565Texture() override {
        if (texture_) {
            if (auto *context = QOpenGLContext::currentContext()) {
                context->functions()->glDeleteTextures(1, &texture_);
            }
            texture_ = 0;
        }
    }

    void setFrame(const std::shared_ptr<VideoItem::Frame> &frame) {
        frame_ = frame;
        dirty_ = true;
    }

    int textureId() const override { return int(texture_); }
    QSize textureSize() const override { return QSize(1280, 720); }
    bool hasAlphaChannel() const override { return false; }
    bool hasMipmaps() const override { return false; }

    bool updateTexture() override {
        if (!dirty_ || !frame_) return false;
        auto *context = QOpenGLContext::currentContext();
        if (!context) return false;
        auto *f = context->functions();
        QElapsedTimer timer;
        timer.start();
        f->glActiveTexture(GL_TEXTURE0);
        if (!texture_) {
            f->glGenTextures(1, &texture_);
            if (!texture_) return false;
            f->glBindTexture(GL_TEXTURE_2D, texture_);
            f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            f->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB565, 1280, 720, 0,
                            GL_RGB, GL_UNSIGNED_SHORT_5_6_5, frame_->pixels.data());
        } else {
            f->glBindTexture(GL_TEXTURE_2D, texture_);
            f->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1280, 720,
                               GL_RGB, GL_UNSIGNED_SHORT_5_6_5, frame_->pixels.data());
        }
        const GLenum error = f->glGetError();
        if (error != GL_NO_ERROR) {
            qWarning() << "QSG RGB565 texture upload failed" << error;
        }
        const double ms = timer.nsecsElapsed() / 1.0e6;
        if (source_) source_->recordQsgUpload(*frame_, ms);
        dirty_ = false;
        return true;
    }

    void bind() override {
        if (dirty_) updateTexture();
        if (texture_) {
            QOpenGLContext::currentContext()->functions()->glBindTexture(GL_TEXTURE_2D, texture_);
        }
    }

private:
    VideoItem *source_ = nullptr;
    std::shared_ptr<VideoItem::Frame> frame_;
    GLuint texture_ = 0;
    bool dirty_ = false;
};

QsgVideoItem::QsgVideoItem(QQuickItem *parent) : QQuickItem(parent) {
    setFlag(ItemHasContents, true);
}

void QsgVideoItem::setSource(VideoItem *source) {
    if (source_ == source) return;
    source_ = source;
    emit sourceChanged();
    update();
}

void QsgVideoItem::requestFrame() {
    update();
}

QSGNode *QsgVideoItem::updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *) {
    auto *node = static_cast<QSGSimpleTextureNode *>(oldNode);
    if (!node) {
        node = new QSGSimpleTextureNode;
        texture_ = new Rgb565Texture(source_);
        node->setTexture(texture_);
        node->setOwnsTexture(true);
        node->setFiltering(QSGTexture::Linear);
        node_ = node;
    }
    if (!source_) return node;

    auto frame = source_->takeDisplayFrame();
    if (frame) {
        texture_->setFrame(frame);
        node->markDirty(QSGNode::DirtyMaterial);
    }
    node->setRect(boundingRect());
    const QRectF vp = source_->sourceViewport();
    node->setSourceRect(QRectF(vp.x() * 1280.0, vp.y() * 720.0,
                               vp.width() * 1280.0, vp.height() * 720.0));
    node->setTextureCoordinatesTransform(QSGSimpleTextureNode::NoTransform);
    return node;
}
