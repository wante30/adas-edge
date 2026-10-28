#include "vehicle_telemetry_status.h"

#include <QDateTime>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QByteArray>
#include <QVariant>

namespace {
QVariant jsonValue(const QVariantMap &map, const char *key)
{
    return map.value(QString::fromLatin1(key));
}

QString textValue(const QVariantMap &map, const char *key, const char *fallback)
{
    const QVariant value = jsonValue(map, key);
    if (!value.isValid() || value.isNull()) return QString::fromLatin1(fallback);
    const QString text = value.toString().trimmed();
    return text.isEmpty() ? QString::fromLatin1(fallback) : text;
}

bool boolValue(const QVariantMap &map, const char *key)
{
    return jsonValue(map, key).toBool();
}

QVariant countValue(const QVariantMap &map, const char *key)
{
    const QVariant value = jsonValue(map, key);
    if (!value.isValid() || value.isNull()) return QVariant();
    bool ok = false;
    const int number = value.toInt(&ok);
    if (!ok || number < 0) return QVariant();
    return number;
}
} // namespace

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
    const bool stale = online && boolValue(snapshot, "sensor_stale");
    const QString source = textValue(snapshot, "lane_source", "none");
    const bool laneValid = online && !stale && source != QStringLiteral("none");
    const bool valid = online && !stale;

    QVariantMap lane;
    lane.insert(QStringLiteral("online"), online);
    lane.insert(QStringLiteral("valid"), laneValid);
    lane.insert(QStringLiteral("stale"), stale);
    lane.insert(QStringLiteral("source"), source);
    lane.insert(QStringLiteral("left_type"), textValue(snapshot, "left_nearest_type", "unknown"));
    lane.insert(QStringLiteral("right_type"), textValue(snapshot, "right_nearest_type", "unknown"));
    lane.insert(QStringLiteral("left_count"), countValue(snapshot, "left_line_count"));
    lane.insert(QStringLiteral("right_count"), countValue(snapshot, "right_line_count"));
    lane.insert(QStringLiteral("simulated"), boolValue(snapshot, "simulated"));

    QVariantMap ego;
    ego.insert(QStringLiteral("online"), online);
    ego.insert(QStringLiteral("valid"), valid);
    ego.insert(QStringLiteral("stale"), stale);
    ego.insert(QStringLiteral("action"), textValue(snapshot, "current_action", "none"));
    ego.insert(QStringLiteral("active"), online && boolValue(snapshot, "active"));
    ego.insert(QStringLiteral("last_action"), textValue(snapshot, "last_action", "none"));
    ego.insert(QStringLiteral("decision"), online && valid
                    ? textValue(snapshot, "decision", "unknown")
                    : QStringLiteral("unknown"));
    ego.insert(QStringLiteral("reason"), online && valid
                    ? textValue(snapshot, "reason", "none")
                    : QStringLiteral("none"));
    ego.insert(QStringLiteral("reason_code"),
               online && valid ? jsonValue(snapshot, "reason_code") : QVariant());
    ego.insert(QStringLiteral("sensor_stale"), stale);
    ego.insert(QStringLiteral("simulated"), boolValue(snapshot, "simulated"));
    const char *const bits[] = {
        "throttle_allow", "brake_allow", "lane_left_allow", "lane_right_allow",
        "throttle_deny", "brake_deny", "lane_left_deny", "lane_right_deny"
    };
    for (const char *key : bits) {
        ego.insert(QString::fromLatin1(key),
                   valid && boolValue(snapshot, key));
    }

    if (online == online_ && valid == valid_ && snapshot == snapshot_ &&
        lane == lane_state_ && ego == ego_state_) return;
    online_ = online;
    valid_ = valid;
    snapshot_ = snapshot;
    lane_state_ = lane;
    ego_state_ = ego;
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
