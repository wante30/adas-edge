#pragma once

#include <QObject>
#include <QString>

class VideoItem;

// Runtime source switch without changing VideoItem's QMetaObject/ABI.
class SourceSwitchController : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString mode READ mode NOTIFY modeChanged)
    Q_PROPERTY(QString message READ message NOTIFY modeChanged)

public:
    SourceSwitchController(double fps, const QString &aiFrameFile, double aiHz,
                           const QString &pcieFile, const QString &localFile,
                           QObject *parent = nullptr);

    void setVideo(VideoItem *video);
    QString mode() const { return mode_; }
    QString message() const { return message_; }

    Q_INVOKABLE void toggle();
    Q_INVOKABLE void switchToPcie();
    Q_INVOKABLE void switchToLocal();

signals:
    void modeChanged();

private:
    void apply(const QString &path, const QString &mode, const QString &label);

    VideoItem *video_ = nullptr;
    double fps_ = 30.0;
    QString aiFrameFile_;
    double aiHz_ = 10.0;
    QString pcieFile_;
    QString localFile_;
    QString mode_ = QStringLiteral("pgv-local");
    QString message_ = QStringLiteral("点击切换 PCIe / 本地回放");
};
