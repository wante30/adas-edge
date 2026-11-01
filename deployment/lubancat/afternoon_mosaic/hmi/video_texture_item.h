#pragma once

#include <QQuickItem>
#include <QSize>

class VideoItem;
class QSGSimpleTextureNode;
class QSGTexture;

// Experimental Scene Graph texture-node video renderer. It reuses the
// existing VideoItem producer/DisplayQueue and never creates a video FBO.
class VideoTextureItem : public QQuickItem {
    Q_OBJECT
    Q_PROPERTY(QQuickItem *source READ source WRITE setSource NOTIFY sourceChanged)

public:
    explicit VideoTextureItem(QQuickItem *parent = nullptr);
    ~VideoTextureItem() override;

    QQuickItem *source() const { return sourceItem_; }
    void setSource(QQuickItem *source);

    Q_INVOKABLE void requestUpdate() { update(); }

signals:
    void sourceChanged();

protected:
    QSGNode *updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *data) override;

private:
    QQuickItem *sourceItem_ = nullptr;
    VideoItem *video_ = nullptr;
    unsigned int textureId_ = 0;
    QSize textureSize_;
    QSGSimpleTextureNode *textureNode_ = nullptr;
    QSGTexture *texture_ = nullptr;
    bool testImageLoaded_ = false;
};
