import re
from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=p.read_text(encoding='utf-8')
start=s.index('            QVariantList boxes;')
end=s.index('            const QByteArray statusData', start)
new=r'''            QVariantList boxes;
            {
                QMutexLocker locker(&ai_metrics_mutex_);
                if (frameId < latest_ai_frame_id_.load() || timestampNs <= 0) {
                    tracks_.clear();
                    next_track_id_ = 1;
                }
                const qint64 detectNs = static_cast<qint64>(timestampNs);
                const auto iou = [](double ax, double ay, double aw, double ah,
                                    double bx, double by, double bw, double bh) {
                    const double ix = std::max(0.0, std::min(ax + aw, bx + bw) - std::max(ax, bx));
                    const double iy = std::max(0.0, std::min(ay + ah, by + bh) - std::max(ay, by));
                    const double intersection = ix * iy;
                    const double unionArea = aw * ah + bw * bh - intersection;
                    return unionArea > 0.0 ? intersection / unionArea : 0.0;
                };

                // Match against the track state at the *detection capture time*,
                // not the state at the time the packet happened to arrive.
                std::vector<std::array<double, 4>> predictedBoxes(tracks_.size());
                for (size_t j = 0; j < tracks_.size(); ++j) {
                    Track &track = tracks_[j];
                    double baseX = track.x;
                    double baseY = track.y;
                    double baseW = track.width;
                    double baseH = track.height;
                    qint64 baseNs = track.lastUpdateNs;
                    for (auto it = track.history.rbegin(); it != track.history.rend(); ++it) {
                        if (it->ns <= detectNs) {
                            baseX = it->x; baseY = it->y; baseW = it->width; baseH = it->height;
                            baseNs = it->ns;
                            break;
                        }
                    }
                    const double dt = std::max<qint64>(0, detectNs - baseNs) / 1e9;
                    const double limit = maxTrackPredictionMs() / 1000.0;
                    const double horizon = std::min(dt, limit);
                    predictedBoxes[j] = {baseX + track.vx * horizon,
                                         baseY + track.vy * horizon,
                                         std::max(3.0, baseW + track.vw * horizon),
                                         std::max(3.0, baseH + track.vh * horizon)};
                }

                std::vector<bool> trackUsed(tracks_.size(), false);
                std::vector<int> detectionTrack(detectionMaps.size(), -1);
                for (int i = 0; i < detectionMaps.size(); ++i) {
                    const QVariantMap detection = detectionMaps.at(i).toMap();
                    const QString className = detection.value(QStringLiteral("className")).toString();
                    const double dx = detection.value(QStringLiteral("x")).toDouble();
                    const double dy = detection.value(QStringLiteral("y")).toDouble();
                    const double dw = detection.value(QStringLiteral("width")).toDouble();
                    const double dh = detection.value(QStringLiteral("height")).toDouble();
                    int best = -1;
                    double bestIou = 0.06;
                    for (int j = 0; j < static_cast<int>(tracks_.size()); ++j) {
                        if (trackUsed[static_cast<size_t>(j)]) continue;
                        Track &track = tracks_[static_cast<size_t>(j)];
                        if (track.className != className) continue;
                        const auto &box = predictedBoxes[static_cast<size_t>(j)];
                        const double overlap = iou(dx, dy, dw, dh, box[0], box[1], box[2], box[3]);
                        // Motion gate: a detection cannot jump more than a
                        // generous fraction of the box size in one transition.
                        const double gate = std::max(dw, dh) * 1.8;
                        const double centerDx = (dx + dw * 0.5) - (box[0] + box[2] * 0.5);
                        const double centerDy = (dy + dh * 0.5) - (box[1] + box[3] * 0.5);
                        if (overlap > bestIou && std::hypot(centerDx, centerDy) <= gate) {
                            bestIou = overlap;
                            best = j;
                        }
                    }
                    detectionTrack[static_cast<size_t>(i)] = best;
                    if (best >= 0) trackUsed[static_cast<size_t>(best)] = true;
                }

                for (int i = 0; i < detectionMaps.size(); ++i) {
                    const int index = detectionTrack[static_cast<size_t>(i)];
                    if (index < 0) continue;
                    const QVariantMap detection = detectionMaps.at(i).toMap();
                    Track &track = tracks_[static_cast<size_t>(index)];
                    const double dx = detection.value(QStringLiteral("x")).toDouble();
                    const double dy = detection.value(QStringLiteral("y")).toDouble();
                    const double dw = detection.value(QStringLiteral("width")).toDouble();
                    const double dh = detection.value(QStringLiteral("height")).toDouble();
                    const auto &predicted = predictedBoxes[static_cast<size_t>(index)];
                    const qint64 baseNs = track.history.empty() ? detectNs : track.history.back().ns;
                    const double dt = std::max(0.03, (detectNs - baseNs) / 1e9);
                    const double measuredVx = (dx - predicted[0]) / dt;
                    const double measuredVy = (dy - predicted[1]) / dt;
                    const double measuredVw = (dw - predicted[2]) / dt;
                    const double measuredVh = (dh - predicted[3]) / dt;
                    track.vx = std::clamp(0.55 * track.vx + 0.45 * measuredVx, -2000.0, 2000.0);
                    track.vy = std::clamp(0.55 * track.vy + 0.45 * measuredVy, -2000.0, 2000.0);
                    track.vw = std::clamp(0.65 * track.vw + 0.35 * measuredVw, -800.0, 800.0);
                    track.vh = std::clamp(0.65 * track.vh + 0.35 * measuredVh, -800.0, 800.0);
                    track.x = dx;
                    track.y = dy;
                    track.width = dw;
                    track.height = dh;
                    track.confidence = detection.value(QStringLiteral("confidence")).toDouble();
                    track.color = detection.value(QStringLiteral("color")).toString();
                    track.lastDetectFrame = frameId;
                    track.lastUpdateFrame = frameId;
                    track.lastDetectNs = detectNs;
                    track.lastUpdateNs = detectNs;
                    track.missed = 0;
                    track.hits += 1;
                    track.state = track.hits >= 2 ? QStringLiteral("confirmed")
                                                  : QStringLiteral("tentative");
                    track.history.push_back({detectNs, dx, dy, dw, dh});
                    while (track.history.size() > 8) track.history.pop_front();
                }

                for (size_t j = 0; j < tracks_.size(); ++j) {
                    if (!trackUsed[j]) {
                        ++tracks_[j].missed;
                        tracks_[j].state = QStringLiteral("predicted");
                    }
                }

                for (int i = 0; i < detectionMaps.size(); ++i) {
                    if (detectionTrack[static_cast<size_t>(i)] >= 0) continue;
                    const QVariantMap detection = detectionMaps.at(i).toMap();
                    Track track;
                    track.id = next_track_id_++;
                    track.classId = detection.value(QStringLiteral("classId")).toInt();
                    track.className = detection.value(QStringLiteral("className")).toString();
                    track.color = detection.value(QStringLiteral("color")).toString();
                    track.x = detection.value(QStringLiteral("x")).toDouble();
                    track.y = detection.value(QStringLiteral("y")).toDouble();
                    track.width = detection.value(QStringLiteral("width")).toDouble();
                    track.height = detection.value(QStringLiteral("height")).toDouble();
                    track.confidence = detection.value(QStringLiteral("confidence")).toDouble();
                    track.lastDetectFrame = frameId;
                    track.lastUpdateFrame = frameId;
                    track.lastDetectNs = detectNs;
                    track.lastUpdateNs = detectNs;
                    track.hits = 1;
                    track.state = QStringLiteral("tentative");
                    track.history.push_back({detectNs, track.x, track.y, track.width, track.height});
                    tracks_.push_back(track);
                }

                constexpr int kMaxMissed = 2;
                tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                             [&](const Track &track) {
                                                 const qint64 age = detectNs - track.lastDetectNs;
                                                 return (track.hits < 2 && track.missed > 0) ||
                                                        track.missed > kMaxMissed ||
                                                        age > 900000000LL;
                                             }),
                              tracks_.end());

                std::vector<QVariantMap> ordered;
                ordered.reserve(tracks_.size());
                for (const Track &track : tracks_) {
                    // High-confidence first detections are shown immediately as
                    // tentative; low-confidence first detections wait for a second
                    // match. No target is created from prediction alone.
                    if (track.hits < 2 && track.confidence < 0.45) continue;
                    if (track.width < 3.0 || track.height < 3.0) continue;
                    QVariantMap item;
                    item.insert(QStringLiteral("x"), track.x);
                    item.insert(QStringLiteral("y"), track.y);
                    item.insert(QStringLiteral("width"), track.width);
                    item.insert(QStringLiteral("height"), track.height);
                    item.insert(QStringLiteral("vx"), track.vx);
                    item.insert(QStringLiteral("vy"), track.vy);
                    item.insert(QStringLiteral("vw"), track.vw);
                    item.insert(QStringLiteral("vh"), track.vh);
                    item.insert(QStringLiteral("frameId"), static_cast<qulonglong>(track.lastDetectFrame));
                    item.insert(QStringLiteral("detectTimestampNs"), static_cast<qulonglong>(track.lastDetectNs));
                    item.insert(QStringLiteral("trackId"), static_cast<qulonglong>(track.id));
                    item.insert(QStringLiteral("missed"), track.missed);
                    item.insert(QStringLiteral("predicted"), track.missed > 0);
                    item.insert(QStringLiteral("state"), track.state);
                    item.insert(QStringLiteral("classId"), track.classId);
                    item.insert(QStringLiteral("className"), track.className);
                    item.insert(QStringLiteral("color"), track.color);
                    item.insert(QStringLiteral("confidence"), track.confidence);
                    item.insert(QStringLiteral("label"),
                                QStringLiteral("%1  %2%")
                                    .arg(track.className.toUpper())
                                    .arg(qRound(track.confidence * 100.0)));
                    ordered.push_back(item);
                }
                std::sort(ordered.begin(), ordered.end(), [](const QVariantMap &a, const QVariantMap &b) {
                    const double aa = a.value(QStringLiteral("width")).toDouble() *
                                      a.value(QStringLiteral("height")).toDouble();
                    const double bb = b.value(QStringLiteral("width")).toDouble() *
                                      b.value(QStringLiteral("height")).toDouble();
                    return aa > bb;
                });
                for (const QVariantMap &item : ordered) boxes.append(item);
            }
'''
p.write_text(s[:start]+new+s[end:],encoding='utf-8')
print('replaced',start,end)
