#pragma once

#include <QObject>
#include <QString>
#include <QUdpSocket>

// Sends only changed, protocol-valid navigation state to the root HUD bridge.
// QML is deliberately not involved: the main HUD must stay independent of the
// optional control panel.
class NavBridge : public QObject {
    Q_OBJECT

public:
    explicit NavBridge(QObject *parent = nullptr);

    void updateNav(const QString &direction, double distanceM);

private:
    QUdpSocket socket_;
    QString lastKey_;
};
