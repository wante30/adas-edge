#pragma once

#include "video_item.h"

#include <QElapsedTimer>
#include <QQuickItem>
#include <QSGDynamicTexture>
#include <QSGSimpleTextureNode>
#include <QSize>

class QsgVideoItem : public QQuickItem {
    Q_OBJECT
    Q_PROPERTY(VideoItem *source READ source WRITE setSource NOTIFY sourceChanged)

public:
    explicit QsgVideoItem(QQuickItem *parent = nullptr);

    VideoItem *source() const { return source_; }
    void setSource(VideoItem *source);

public slots:
    void requestFrame();

signals:
    void sourceChanged();

protected:
    QSGNode *updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *data) override;

private:
    class Rgb565Texture;

    VideoItem *source_ = nullptr;
    QSGSimpleTextureNode *node_ = nullptr;
    Rgb565Texture *texture_ = nullptr;
};
