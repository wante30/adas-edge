#include "nav_bridge.h"

#include <QHostAddress>
#include <QJsonDocument>
#include <QJsonObject>
#include <QtMath>

namespace {
constexpr quint16 kBridgePort = 38262;
}

NavBridge::NavBridge(QObject *parent) : QObject(parent) {
    socket_.bind(QHostAddress(QHostAddress::LocalHost), 0);
}

void NavBridge::updateNav(const QString &direction, double distanceM) {
    const QString dir = direction.trimmed().toLower();
    QString mapped;
    if (dir == QStringLiteral("forward") || dir == QStringLiteral("straight"))
        mapped = QStringLiteral("straight");
    else if (dir == QStringLiteral("left"))
        mapped = QStringLiteral("left");
    else if (dir == QStringLiteral("right"))
        mapped = QStringLiteral("right");

    const bool valid = !mapped.isEmpty() && qIsFinite(distanceM) && distanceM >= 0.0;
    const int metres = valid ? qBound(0, qRound(distanceM), 65535) : 0;
    const QString key = valid ? mapped + QLatin1Char(':') + QString::number(metres)
                              : QStringLiteral("off");
    if (key == lastKey_)
        return;
    lastKey_ = key;

    QJsonObject command;
    command.insert(QStringLiteral("type"), QStringLiteral("nav"));
    command.insert(QStringLiteral("action"), QStringLiteral("apply"));
    command.insert(QStringLiteral("enabled"), valid);
    command.insert(QStringLiteral("direction"), valid ? mapped : QStringLiteral("straight"));
    command.insert(QStringLiteral("distance_m"), metres);
    const QByteArray packet = QJsonDocument(command).toJson(QJsonDocument::Compact);
    socket_.writeDatagram(packet, QHostAddress(QHostAddress::LocalHost), kBridgePort);
}
