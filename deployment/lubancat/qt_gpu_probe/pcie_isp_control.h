#pragma once

#include <QObject>
#include <QString>
#include <QUdpSocket>

class PcieIspControl : public QObject {
    Q_OBJECT

public:
    explicit PcieIspControl(QObject *parent = nullptr);

    Q_INVOKABLE void query();
    Q_INVOKABLE void applyIspReset();
    Q_INVOKABLE void applyIspMaster(bool enabled);
    Q_INVOKABLE void applyAlgo(bool dehaze, bool gaussian, bool median, bool sobel);

signals:
    void replyArrived(const QString &summary, const QString &status, const QString &mode);

private:
    void send(const QJsonObject &object);
    QUdpSocket socket_;
    QString summary_ = QStringLiteral("按 I 打开 ISP 控制，先 READ");
    QString status_ = QStringLiteral("OFFLINE");
    QString mode_ = QStringLiteral("UNKNOWN");
};
