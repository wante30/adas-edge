#include "hud_control.h"

#include <QDateTime>
#include <QHostAddress>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkDatagram>
#include <algorithm>

namespace {
constexpr quint16 kBridgePort = 38262;
constexpr qint64 kOfflineAfterNs = 2500000000LL;
}

HudControl::HudControl(QObject *parent) : QObject(parent) {
    socket_.bind(QHostAddress(QHostAddress::LocalHost), 0);
    connect(&socket_, &QUdpSocket::readyRead, this, &HudControl::readPending);
    connect(&ping_timer_, &QTimer::timeout, this, &HudControl::ping);
    ping_timer_.start(2000);
    connect(&age_timer_, &QTimer::timeout, this, &HudControl::checkAge);
    age_timer_.start(500);
    ping();
}

HudControl::~HudControl() = default;

void HudControl::sendCommand(const QJsonObject &object) {
    const QByteArray packet = QJsonDocument(object).toJson(QJsonDocument::Compact);
    if (socket_.writeDatagram(packet, QHostAddress(QHostAddress::LocalHost), kBridgePort) != packet.size()) {
        setStatus(QStringLiteral("OFFLINE"), QStringLiteral("bridge send failed"));
    }
}

void HudControl::ping() {
    QJsonObject object;
    object.insert(QStringLiteral("type"), QStringLiteral("status"));
    sendCommand(object);
}

void HudControl::sendPerception(bool perception_enable, int traffic, bool pedestrian,
                                bool static_target, bool overspeed, int lane_left,
                                int lane_right, bool lka_warning, bool time_enable,
                                bool legacy_enable, int legacy_state) {
    QJsonObject object;
    object.insert(QStringLiteral("type"), QStringLiteral("set"));
    object.insert(QStringLiteral("perception_enable"), perception_enable);
    object.insert(QStringLiteral("traffic"), std::clamp(traffic, 0, 2));
    object.insert(QStringLiteral("pedestrian"), pedestrian);
    object.insert(QStringLiteral("static_target"), static_target);
    object.insert(QStringLiteral("overspeed"), overspeed);
    object.insert(QStringLiteral("lane_left"), std::clamp(lane_left, 0, 2));
    object.insert(QStringLiteral("lane_right"), std::clamp(lane_right, 0, 2));
    object.insert(QStringLiteral("lka_warning"), lka_warning);
    object.insert(QStringLiteral("time_enable"), time_enable);
    object.insert(QStringLiteral("legacy_enable"), legacy_enable);
    object.insert(QStringLiteral("legacy_state"), legacy_state & 0x1ff);
    sendCommand(object);
}

void HudControl::sendNavigation(bool enabled, const QString &direction, int distanceM) {
    QJsonObject object;
    object.insert(QStringLiteral("type"), QStringLiteral("nav"));
    object.insert(QStringLiteral("action"), QStringLiteral("apply"));
    object.insert(QStringLiteral("enabled"), enabled);
    object.insert(QStringLiteral("direction"), direction);
    object.insert(QStringLiteral("distance_m"), std::clamp(distanceM, 0, 65535));
    sendCommand(object);
}

void HudControl::queryNav() {
    QJsonObject object;
    object.insert(QStringLiteral("type"), QStringLiteral("nav"));
    object.insert(QStringLiteral("action"), QStringLiteral("status"));
    sendCommand(object);
}

void HudControl::queryProtocol() {
    QJsonObject object;
    object.insert(QStringLiteral("type"), QStringLiteral("protocol"));
    sendCommand(object);
}

void HudControl::queryIsp() {
    QJsonObject object;
    object.insert(QStringLiteral("type"), QStringLiteral("isp"));
    object.insert(QStringLiteral("action"), QStringLiteral("query"));
    sendCommand(object);
}

void HudControl::readPending() {
    while (socket_.hasPendingDatagrams()) {
        const QNetworkDatagram datagram = socket_.receiveDatagram();
        const QJsonObject reply = QJsonDocument::fromJson(datagram.data()).object();
        if (reply.isEmpty()) continue;
        bridge_online_ = true;
        last_reply_ns_ = QDateTime::currentMSecsSinceEpoch() * 1000000LL;
        last_applied_seq_ = reply.value(QStringLiteral("ext_applied")).toInt(last_applied_seq_);
        last_error_ = reply.value(QStringLiteral("error")).toInt(0);
        const bool ok = reply.value(QStringLiteral("ok")).toBool(false);
        const QString type = reply.value(QStringLiteral("type")).toString();
        if (type == QStringLiteral("nav")) {
            nav_status_ = ok ? QStringLiteral("OK") : QStringLiteral("UNCONFIRMED");
        } else if (type == QStringLiteral("isp")) {
            isp_status_ = ok ? QStringLiteral("OK") : QStringLiteral("UNSUPPORTED");
        } else if (type == QStringLiteral("protocol")) {
            protocol_summary_ = reply.value(QStringLiteral("summary")).toString();
            protocol_lines_.clear();
            for (const QString &line : protocol_summary_.split(QLatin1Char('\n'))) {
                if (line.isEmpty()) continue;
                for (int i = 0; i < line.size(); i += 110)
                    protocol_lines_.append(line.mid(i, 110));
            }
        }
        QString status = ok ? QStringLiteral("ONLINE") : QStringLiteral("REJECT");
        if (reply.value(QStringLiteral("type")).toString() == QStringLiteral("status"))
            status = QStringLiteral("ONLINE");
        setStatus(status, QString::fromUtf8(datagram.data()),
                  last_applied_seq_, last_error_);
    }
}

void HudControl::checkAge() {
    if (!bridge_online_) return;
    const qint64 now = QDateTime::currentMSecsSinceEpoch() * 1000000LL;
    if (now - last_reply_ns_ > kOfflineAfterNs) {
        bridge_online_ = false;
        setStatus(QStringLiteral("OFFLINE"), QStringLiteral("bridge heartbeat lost"));
    }
}

void HudControl::setStatus(const QString &status, const QString &reply,
                           int applied_seq, int error) {
    bridge_status_ = status;
    if (!reply.isEmpty() && reply != last_reply_) last_reply_ = reply;
    if (applied_seq >= 0) last_applied_seq_ = applied_seq;
    if (error >= 0) last_error_ = error;
    emit statusChanged();
}
