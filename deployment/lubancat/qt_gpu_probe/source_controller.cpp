#include "source_controller.h"
#include "video_item.h"
#include <QDebug>

SourceSwitchController::SourceSwitchController(double fps, const QString &aiFrameFile, double aiHz,
                                               const QString &pcieFile, const QString &localFile,
                                               QObject *parent)
    : QObject(parent), fps_(fps), aiFrameFile_(aiFrameFile), aiHz_(aiHz),
      pcieFile_(pcieFile), localFile_(localFile) {}

void SourceSwitchController::setVideo(VideoItem *video) {
    video_ = video;
    if (video_) mode_ = video_->sourceMode();
    emit modeChanged();
}

void SourceSwitchController::apply(const QString &path, const QString &mode) {
    qWarning() << "SourceSwitch apply" << path << mode << "video" << (video_ != nullptr);
    if (!video_ || path.isEmpty()) return;
    video_->startPgvProducer(fps_, path, aiFrameFile_, aiHz_, mode);
    mode_ = mode;
    emit modeChanged();
}

void SourceSwitchController::toggle() {
    qWarning() << "SourceSwitch toggle mode" << mode_;
    if (mode_ == QStringLiteral("pgv-live")) switchToLocal();
    else switchToPcie();
}

void SourceSwitchController::switchToPcie() { apply(pcieFile_, QStringLiteral("pgv-live")); }
void SourceSwitchController::switchToLocal() { apply(localFile_, QStringLiteral("pgv-local")); }
