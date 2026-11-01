#pragma once

#include <QObject>
#include <QString>
#include <QTimer>
#include <QVariantMap>

// Read-only, optional JSON-file consumer for LVT1. No BAR/MMIO/DMA access.
// FPGA telemetry is NOT itself a physical actuation confirmation.
class VehicleTelemetryStatus : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool online READ online NOTIFY statusChanged)
    Q_PROPERTY(bool valid READ valid NOTIFY statusChanged)
    Q_PROPERTY(QVariantMap snapshot READ snapshot NOTIFY statusChanged)
    Q_PROPERTY(QVariantMap laneState READ laneState NOTIFY statusChanged)
    Q_PROPERTY(QVariantMap egoState READ egoState NOTIFY statusChanged)
public:
    explicit VehicleTelemetryStatus(QObject *parent = nullptr);
    bool online() const { return online_; }
    bool valid() const { return valid_; }
    QVariantMap snapshot() const { return snapshot_; }
    QVariantMap laneState() const { return lane_state_; }
    QVariantMap egoState() const { return ego_state_; }
signals:
    void statusChanged();
private:
    void refresh();
    void publish(bool online, const QVariantMap &snapshot);
    QString path_;
    QTimer poll_;
    bool online_ = false;
    bool valid_ = false;
    QVariantMap snapshot_;
    QVariantMap lane_state_;
    QVariantMap ego_state_;
};
