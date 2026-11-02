#pragma once

#include <QObject>
#include <QString>
#include <QTimer>
#include <QVariantMap>

// Read-only, optional JSON-file consumer for LVT1. No BAR/MMIO/DMA access.
// FPGA telemetry is NOT itself a physical actuation confirmation.
class VehicleTelemetryStatus final : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool online READ online NOTIFY statusChanged)
    Q_PROPERTY(QVariantMap snapshot READ snapshot NOTIFY statusChanged)
public:
    explicit VehicleTelemetryStatus(QObject *parent = nullptr);
    bool online() const { return online_; }
    QVariantMap snapshot() const { return snapshot_; }
signals:
    void statusChanged();
private:
    void refresh();
    void publish(bool online, const QVariantMap &snapshot);
    QString path_;
    QTimer poll_;
    bool online_ = false;
    QVariantMap snapshot_;
};
