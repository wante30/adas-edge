#include "source_controller.h"
#include "video_item.h"
#include <QDebug>
#include <QFileInfo>

SourceSwitchController::SourceSwitchController(double fps, const QString &aiFrameFile, double aiHz,
                                               const QString &pcieFile, const QString &localFile,
                                               QObject *parent)
    : QObject(parent), fps_(fps), aiFrameFile_(aiFrameFile), aiHz_(aiHz),
      pcieFile_(pcieFile), localFile_(localFile) {}

void SourceSwitchController::setVideo(VideoItem *video) {
    video_ = video;
    if (video_) mode_ = video_->sourceMode();
    message_ = QStringLiteral("点击切换 PCIe / 本地回放");
    emit modeChanged();
}

void SourceSwitchController::apply(const QString &path, const QString &mode, const QString &label) {
    qWarning() << "SourceSwitch apply" << path << mode << "video" << (video_ != nullptr);
    if (!video_ || path.isEmpty()) {
        message_ = QStringLiteral("切换失败：视频路径为空");
        emit modeChanged();
        return;
    }
    if (!QFileInfo::exists(path)) {
        message_ = QStringLiteral("%1未准备：%2").arg(label, path);
        emit modeChanged();
        return;
    }
    video_->startPgvProducer(fps_, path, aiFrameFile_, aiHz_, mode);
    mode_ = mode;
    message_ = QStringLiteral("已切换：%1").arg(label);
    emit modeChanged();
}

void SourceSwitchController::toggle() {
    qWarning() << "SourceSwitch toggle mode" << mode_;
    if (mode_ == QStringLiteral("pgv-live")) switchToLocal();
    else switchToPcie();
}

void SourceSwitchController::switchToPcie() {
    apply(pcieFile_, QStringLiteral("pgv-live"), QStringLiteral("PCIe"));
}

void SourceSwitchController::switchToLocal() {
    apply(localFile_, QStringLiteral("pgv-local"), QStringLiteral("本地回放"));
}
