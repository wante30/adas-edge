#pragma once

#include <QJsonObject>
#include <QVariantList>
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
    Q_PROPERTY(QString navStatus READ navStatus NOTIFY statusChanged)
    Q_PROPERTY(QString ispStatus READ ispStatus NOTIFY statusChanged)
    Q_PROPERTY(QString protocolSummary READ protocolSummary NOTIFY statusChanged)
    Q_PROPERTY(QVariantList protocolLines READ protocolLines NOTIFY statusChanged)

public:
    explicit HudControl(QObject *parent = nullptr);
    ~HudControl() override;

    QString bridgeStatus() const { return bridge_status_; }
    QString lastReply() const { return last_reply_; }
    bool bridgeOnline() const { return bridge_online_; }
    int lastAppliedSeq() const { return last_applied_seq_; }
    int lastError() const { return last_error_; }
    QString navStatus() const { return nav_status_; }
    QString ispStatus() const { return isp_status_; }
    QString protocolSummary() const { return protocol_summary_; }
    QVariantList protocolLines() const { return protocol_lines_; }

    Q_INVOKABLE void sendPerception(bool perception_enable, int traffic,
                                    bool pedestrian, bool static_target,
                                    bool overspeed, int lane_left, int lane_right,
                                    bool lka_warning, bool time_enable,
                                    bool legacy_enable, int legacy_state = 0x101);
    Q_INVOKABLE void ping();
    Q_INVOKABLE void sendNavigation(bool enabled, const QString &direction, int distanceM);
    Q_INVOKABLE void queryNav();
    Q_INVOKABLE void queryIsp();
    Q_INVOKABLE void queryProtocol();

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
    QString nav_status_ = QStringLiteral("UNKNOWN");
    QString isp_status_ = QStringLiteral("UNKNOWN");
    QString protocol_summary_ = QStringLiteral("按 T 打开后点击 READ ALL");
    QVariantList protocol_lines_;
};
