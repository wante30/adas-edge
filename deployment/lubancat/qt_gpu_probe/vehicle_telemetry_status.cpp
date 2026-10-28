#include "vehicle_telemetry_status.h"

#include <QDateTime>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QByteArray>

VehicleTelemetryStatus::VehicleTelemetryStatus(QObject *parent)
    : QObject(parent),
      path_(qEnvironmentVariable("LVT1_STATUS_FILE",
                                "/dev/shm/adas_vehicle_status.json")) {
    poll_.setInterval(150);  // UI status only; not a real-time control path
    connect(&poll_, &QTimer::timeout, this, &VehicleTelemetryStatus::refresh);
    poll_.start();
    refresh();
}

void VehicleTelemetryStatus::publish(bool online, const QVariantMap &snapshot) {
    if (online == online_ && snapshot == snapshot_) return;
    online_ = online;
    snapshot_ = snapshot;
    emit statusChanged();
}

void VehicleTelemetryStatus::refresh() {
    QFileInfo info(path_);
    // FPGA freshness is separately carried in LVT1; this is the RK-side
    // local file freshness, preventing old online:true from surviving a crash.
    const qint64 ageMs = QDateTime::currentMSecsSinceEpoch()
                        - info.lastModified().toMSecsSinceEpoch();
    if (!info.isFile() || ageMs < 0 || ageMs > 400 || info.size() > 8192) {
        publish(false, QVariantMap());
        return;
    }
    QFile file(path_);
    if (!file.open(QIODevice::ReadOnly)) {
        publish(false, QVariantMap());
        return;
    }
    const QByteArray raw = file.read(8193);
    if (raw.size() > 8192) {
        publish(false, QVariantMap());
        return;
    }
    QJsonParseError error;
    const QJsonDocument doc = QJsonDocument::fromJson(raw, &error);
    if (error.error != QJsonParseError::NoError || !doc.isObject()) {
        publish(false, QVariantMap());
        return;
    }
    const QJsonObject obj = doc.object();
    if (obj.value("schema").toString() !=
            QStringLiteral("pango.rk3566.vehicle_status.v1") ||
        !obj.value("online").toBool(false)) {
        publish(false, QVariantMap());
        return;
    }
    publish(true, obj.toVariantMap());
}
