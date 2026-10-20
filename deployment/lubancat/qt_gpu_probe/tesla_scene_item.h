#pragma once

#include <QMutex>
#include <QQuickFramebufferObject>
#include <QVariantList>
#include <atomic>

class TeslaSceneItem : public QQuickFramebufferObject {
    Q_OBJECT
    Q_PROPERTY(QVariantList boxes READ boxes WRITE setBoxes NOTIFY boxesChanged)
    Q_PROPERTY(bool online READ online WRITE setOnline NOTIFY onlineChanged)
    Q_PROPERTY(bool demoMode READ demoMode WRITE setDemoMode NOTIFY demoModeChanged)
    Q_PROPERTY(QString sourceKey READ sourceKey WRITE setSourceKey NOTIFY sourceKeyChanged)
    Q_PROPERTY(double sceneFps READ sceneFps NOTIFY sceneFpsChanged)

public:
    explicit TeslaSceneItem(QQuickItem *parent = nullptr);

    QQuickFramebufferObject::Renderer *createRenderer() const override;

    QVariantList boxes() const;
    void setBoxes(const QVariantList &boxes);
    bool online() const { return online_.load(); }
    void setOnline(bool online);
    bool demoMode() const { return demo_mode_.load(); }
    void setDemoMode(bool demo);
    QString sourceKey() const;
    void setSourceKey(const QString &key);
    double sceneFps() const { return scene_fps_.load(); }
    Q_INVOKABLE void requestUpdate() { update(); }
    void noteSceneFps(double fps) { scene_fps_.store(fps); emit sceneFpsChanged(); }

signals:
    void boxesChanged();
    void onlineChanged();
    void demoModeChanged();
    void sourceKeyChanged();
    void sceneFpsChanged();

private:
    mutable QMutex mutex_;
    QVariantList boxes_;
    QString source_key_;
    std::atomic<bool> online_{false};
    std::atomic<bool> demo_mode_{false};
    std::atomic<double> scene_fps_{0.0};
};
