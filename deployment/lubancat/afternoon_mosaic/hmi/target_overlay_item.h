#pragma once
#include <QPointer>
#include <QQuickItem>
#include <QVariantList>

// One immutable result snapshot replaces hundreds of per-delegate JS bindings.
class TargetOverlayItem : public QQuickItem {
    Q_OBJECT
    Q_PROPERTY(QObject *videoSource READ videoSource WRITE setVideoSource)
    Q_PROPERTY(QVariantList boxes READ boxes WRITE setBoxes)
    Q_PROPERTY(qint64 displayTimestampNs MEMBER display_ns_)
    Q_PROPERTY(qulonglong displayFrameId MEMBER display_frame_)
    Q_PROPERTY(double sourceFps MEMBER source_fps_)
    Q_PROPERTY(int predictionMs MEMBER prediction_ms_)
    Q_PROPERTY(int trackingMode MEMBER tracking_mode_)
    Q_PROPERTY(QString fontFamily MEMBER font_family_)
public:
    QObject *videoSource() const { return video_source_; }
    void setVideoSource(QObject *value);
    explicit TargetOverlayItem(QQuickItem *parent=nullptr);
    QVariantList boxes() const { return boxes_; }
    void setBoxes(const QVariantList &value);
protected:
    QSGNode *updatePaintNode(QSGNode *, UpdatePaintNodeData *) override;
private:
    QPointer<QObject> video_source_;
    QVariantList boxes_;
    qint64 display_ns_=0;
    qulonglong display_frame_=0;
    double source_fps_=60;
    int prediction_ms_=250, tracking_mode_=2;
    QString font_family_;
};
