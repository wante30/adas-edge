#pragma once

#include <QJsonObject>
#include <QObject>
#include <QTimer>
#include <QUdpSocket>

// Client for the root-owned BAR1 HUD bridge. Qt itself never maps PCIe BAR1;
// it sends validated perception snapshots to 127.0.0.1:38262.
class HudControl : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString bridgeStatus READ bridgeStatus NOTIFY statusChanged)
    Q_PROPERTY(QString lastReply READ lastReply NOTIFY statusChanged)
    Q_PROPERTY(bool bridgeOnline READ bridgeOnline NOTIFY statusChanged)
    Q_PROPERTY(int lastAppliedSeq READ lastAppliedSeq NOTIFY statusChanged)
    Q_PROPERTY(int lastError READ lastError NOTIFY statusChanged)

public:
    explicit HudControl(QObject *parent = nullptr);
    ~HudControl() override;

    QString bridgeStatus() const { return bridge_status_; }
    QString lastReply() const { return last_reply_; }
    bool bridgeOnline() const { return bridge_online_; }
    int lastAppliedSeq() const { return last_applied_seq_; }
    int lastError() const { return last_error_; }

    Q_INVOKABLE void sendPerception(bool perception_enable, int traffic,
                                    bool pedestrian, bool static_target,
                                    bool overspeed, int lane_left, int lane_right,
                                    bool lka_warning, bool time_enable,
                                    bool legacy_enable, int legacy_state = 0x101);
    Q_INVOKABLE void ping();

signals:
    void statusChanged();

private slots:
    void readPending();
    void checkAge();

private:
    void sendCommand(const QJsonObject &object);
    void setStatus(const QString &status, const QString &reply = QString(),
                   int applied_seq = -1, int error = -1);

    QUdpSocket socket_;
    QTimer ping_timer_;
    QTimer age_timer_;
    qint64 last_reply_ns_ = 0;
    QString bridge_status_ = QStringLiteral("OFFLINE");
    QString last_reply_;
    bool bridge_online_ = false;
    int last_applied_seq_ = 0;
    int last_error_ = 0;
};
