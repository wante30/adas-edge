#pragma once

#include <QMutex>
#include <QQuickFramebufferObject>
#include <QVariantList>
#include <QVariantMap>
#include <atomic>

class TeslaSceneItem : public QQuickFramebufferObject {
    Q_OBJECT
    Q_PROPERTY(QVariantList boxes READ boxes WRITE setBoxes NOTIFY boxesChanged)
    Q_PROPERTY(QVariantMap laneState READ laneState WRITE setLaneState NOTIFY laneStateChanged)
    Q_PROPERTY(QString displayView READ displayView WRITE setDisplayView NOTIFY displayViewChanged)
    Q_PROPERTY(int roadQuadrant READ roadQuadrant WRITE setRoadQuadrant NOTIFY roadQuadrantChanged)
    Q_PROPERTY(bool online READ online WRITE setOnline NOTIFY onlineChanged)
    Q_PROPERTY(bool demoMode READ demoMode WRITE setDemoMode NOTIFY demoModeChanged)
    Q_PROPERTY(QString sourceKey READ sourceKey WRITE setSourceKey NOTIFY sourceKeyChanged)
    Q_PROPERTY(qulonglong sourceFrameId READ sourceFrameId WRITE setSourceFrameId NOTIFY sourceFrameIdChanged)
    Q_PROPERTY(qint64 sourceTimestampNs READ sourceTimestampNs WRITE setSourceTimestampNs NOTIFY sourceTimestampNsChanged)
    Q_PROPERTY(double sceneFps READ sceneFps NOTIFY sceneFpsChanged)

public:
    explicit TeslaSceneItem(QQuickItem *parent = nullptr);

    QQuickFramebufferObject::Renderer *createRenderer() const override;

    QVariantList boxes() const;
    void setBoxes(const QVariantList &boxes);
    QVariantMap laneState() const;
    void setLaneState(const QVariantMap &state);
    QString displayView() const;
    void setDisplayView(const QString &view);
    int roadQuadrant() const { return road_quadrant_.load(); }
    void setRoadQuadrant(int quadrant);
    bool online() const { return online_.load(); }
    void setOnline(bool online);
    bool demoMode() const { return demo_mode_.load(); }
    void setDemoMode(bool demo);
    QString sourceKey() const;
    void setSourceKey(const QString &key);
    qulonglong sourceFrameId() const { return source_frame_id_.load(); }
    void setSourceFrameId(qulonglong id);
    qint64 sourceTimestampNs() const { return source_timestamp_ns_.load(); }
    void setSourceTimestampNs(qint64 ns);
    double sceneFps() const { return scene_fps_.load(); }
    Q_INVOKABLE void requestUpdate() { update(); }
    void noteSceneFps(double fps) { scene_fps_.store(fps); emit sceneFpsChanged(); }

signals:
    void boxesChanged();
    void laneStateChanged();
    void displayViewChanged();
    void roadQuadrantChanged();
    void onlineChanged();
    void demoModeChanged();
    void sourceKeyChanged();
    void sourceFrameIdChanged();
    void sourceTimestampNsChanged();
    void sceneFpsChanged();

private:
    mutable QMutex mutex_;
    QVariantList boxes_;
    QVariantMap lane_state_;
    QString display_view_;
    QString source_key_;
    std::atomic<int> road_quadrant_{1};
    std::atomic<bool> online_{false};
    std::atomic<bool> demo_mode_{false};
    std::atomic<double> scene_fps_{0.0};
    std::atomic<qulonglong> source_frame_id_{0};
    std::atomic<qint64> source_timestamp_ns_{0};
};
