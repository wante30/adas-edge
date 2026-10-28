from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\source_controller.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''void SourceSwitchController::apply(const QString &path, const QString &mode) {
    if (!video_ || path.isEmpty()) return;''','''void SourceSwitchController::apply(const QString &path, const QString &mode) {
    qWarning() << "SourceSwitch apply" << path << mode << "video" << (video_ != nullptr);
    if (!video_ || path.isEmpty()) return;''',1)
s=s.replace('''void SourceSwitchController::toggle() {
    if (mode_ == QStringLiteral("pgv-live")) switchToLocal();''','''void SourceSwitchController::toggle() {
    qWarning() << "SourceSwitch toggle mode" << mode_;
    if (mode_ == QStringLiteral("pgv-live")) switchToLocal();''',1)
s=s.replace('#include "video_item.h"','#include "video_item.h"\n#include <QDebug>',1)
p.write_text(s,encoding='utf-8')
print('logs')
