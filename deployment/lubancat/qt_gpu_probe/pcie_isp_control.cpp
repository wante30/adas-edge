#include "pcie_isp_control.h"

#include <QHostAddress>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkDatagram>

namespace {
constexpr quint16 kPort = 38264;
constexpr quint32 kTileReset[4][4] = {
    {0x40000000u, 0x21840410u, 0x00000001u, 0x00401004u},
    {0x40000000u, 0x21840410u, 0x00000001u, 0x00401004u},
    {0x40000000u, 0x21840410u, 0x00000001u, 0x00401004u},
    {0x40000000u, 0x21840410u, 0x00000001u, 0x00401004u},
};
}

PcieIspControl::PcieIspControl(QObject *parent) : QObject(parent) {
    socket_.bind(QHostAddress(QHostAddress::LocalHost), 0);
    connect(&socket_, &QUdpSocket::readyRead, this, [this] {
        while (socket_.hasPendingDatagrams()) {
            const QByteArray data = socket_.receiveDatagram().data();
            const QJsonObject reply = QJsonDocument::fromJson(data).object();
            if (reply.isEmpty()) continue;
            status_ = reply.value(QStringLiteral("state")).toString(
                reply.value(QStringLiteral("ok")).toBool() ? QStringLiteral("OK")
                                                           : QStringLiteral("REJECT"));
            summary_ = QString::fromUtf8(QJsonDocument(reply).toJson(QJsonDocument::Indented));
            emit replyChanged();
        }
    });
}

void PcieIspControl::send(const QJsonObject &object) {
    socket_.writeDatagram(QJsonDocument(object).toJson(QJsonDocument::Compact),
                          QHostAddress(QHostAddress::LocalHost), kPort);
}

void PcieIspControl::query() {
    status_ = QStringLiteral("QUERY...");
    emit replyChanged();
    QJsonObject o;
    o.insert(QStringLiteral("type"), QStringLiteral("query"));
    send(o);
}

void PcieIspControl::applyIspReset() {
    QJsonArray tiles;
    for (const auto &tile : kTileReset) {
        QJsonArray values;
        for (quint32 value : tile)
            values.append(QString::number(value, 16).rightJustified(8, QLatin1Char('0')).prepend(QStringLiteral("0x")));
        tiles.append(values);
    }
    QJsonObject o;
    o.insert(QStringLiteral("type"), QStringLiteral("apply_isp"));
    o.insert(QStringLiteral("global"), QStringLiteral("0x1e"));
    o.insert(QStringLiteral("tiles"), tiles);
    status_ = QStringLiteral("APPLY ISP...");
    emit replyChanged();
    send(o);
}

void PcieIspControl::applyIspMaster(bool enabled) {
    QJsonArray tiles;
    for (const auto &tile : kTileReset) {
        QJsonArray values;
        for (quint32 value : tile)
            values.append(QString::number(value, 16).rightJustified(8, QLatin1Char('0')).prepend(QStringLiteral("0x")));
        tiles.append(values);
    }
    QJsonObject o;
    o.insert(QStringLiteral("type"), QStringLiteral("apply_isp"));
    o.insert(QStringLiteral("global"), enabled ? QStringLiteral("0x1f") : QStringLiteral("0x1e"));
    o.insert(QStringLiteral("tiles"), tiles);
    status_ = enabled ? QStringLiteral("ISP MASTER ON") : QStringLiteral("ISP MASTER OFF");
    emit replyChanged();
    send(o);
}

void PcieIspControl::applyAlgo(bool dehaze, bool gaussian, bool median, bool sobel) {
    quint64 stage = 0;
    if (dehaze) stage |= 1ULL << 56;
    if (gaussian) stage |= 1ULL << 57;
    if (median) stage |= 1ULL << 58;
    if (sobel) stage |= 1ULL << 59;
    const quint64 cfg = stage | (0x0fULL << 48) | (230ULL << 40) |
                        (25ULL << 32) | (220ULL << 24) | (1ULL << 16) | 128ULL;
    QJsonObject o;
    o.insert(QStringLiteral("type"), QStringLiteral("apply_algo"));
    o.insert(QStringLiteral("value"), QStringLiteral("0x") + QString::number(cfg, 16));
    status_ = QStringLiteral("APPLY ALGO...");
    emit replyChanged();
    send(o);
}
