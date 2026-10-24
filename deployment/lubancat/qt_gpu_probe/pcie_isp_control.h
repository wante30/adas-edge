#pragma once

#include <QObject>
#include <QString>
#include <QUdpSocket>

class PcieIspControl : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString summary READ summary NOTIFY replyChanged)
    Q_PROPERTY(QString status READ status NOTIFY replyChanged)

public:
    explicit PcieIspControl(QObject *parent = nullptr);
    QString summary() const { return summary_; }
    QString status() const { return status_; }

    Q_INVOKABLE void query();
    Q_INVOKABLE void applyIspReset();
    Q_INVOKABLE void applyIspMaster(bool enabled);
    Q_INVOKABLE void applyAlgo(bool dehaze, bool gaussian, bool median, bool sobel);

signals:
    void replyChanged();

private:
    void send(const QJsonObject &object);
    QUdpSocket socket_;
    QString summary_ = QStringLiteral("按 I 打开 ISP 控制，先 READ");
    QString status_ = QStringLiteral("OFFLINE");
};
