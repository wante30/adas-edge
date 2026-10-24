#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "hud_control.h"
#include "nav_bridge.h"
#include "pcie_isp_control.h"
#include "video_item.h"
#include "tesla_scene_item.h"

#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QSet>
#include <QStringList>
#include <QQuickView>
#include <QSurfaceFormat>
#include <QQmlContext>
#include <QTimer>
#include <QUrl>
#include <QDebug>
#include <QGuiApplication>
#include <algorithm>
#include <QElapsedTimer>
#include <algorithm>
#include <cstdlib>
#include <link.h>

static int intArg(const QStringList &args, const QString &name, int fallback) {
    const int i = args.indexOf(name);
    if (i >= 0 && i + 1 < args.size()) return args.at(i + 1).toInt();
    return fallback;
}
static QString stringArg(const QStringList &args, const QString &name) {
    const int i = args.indexOf(name);
    return i >= 0 && i + 1 < args.size() ? args.at(i + 1) : QString();
}

static QStringList readReplayManifest(const QString &path) {
    QStringList frames;
    if (path.isEmpty()) return frames;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qCritical().noquote() << "Cannot open replay manifest" << path << file.errorString();
        return {};
    }
    while (!file.atEnd()) {
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (line.isEmpty() || line.startsWith('#')) continue;
        const int first = line.indexOf(QChar(' '));
        const int second = first < 0 ? -1 : line.indexOf(QChar(' '), first + 1);
        if (second < 0 || second + 1 >= line.size()) {
            qCritical().noquote() << "Invalid replay manifest row" << line;
            return {};
        }
        const QString framePath = line.mid(second + 1).trimmed();
        if (!QFileInfo::exists(framePath)) {
            qCritical().noquote() << "Replay frame is missing" << framePath;
            return {};
        }
        frames.append(framePath);
    }
    return frames;
}

static int collectGraphicsLibrary(struct dl_phdr_info *info, size_t, void *data) {
    if (!info->dlpi_name || !*info->dlpi_name) return 0;
    auto *paths = static_cast<QSet<QString> *>(data);
    const QString path = QString::fromLocal8Bit(info->dlpi_name);
    if (path.contains("libmali.so") || path.contains("libEGL") ||
        path.contains("libGLES") || path.contains("libqxcb") ||
        path.contains("libQt5Quick")) {
        paths->insert(path);
    }
    return 0;
}

static PcieIspControl *g_ispControl = nullptr;
static QObject *ispControlSingleton(QQmlEngine *, QJSEngine *) { return g_ispControl; }

static QJsonArray loadedGraphicsLibraries() {
    QJsonArray libraries;
    QSet<QString> paths;
    dl_iterate_phdr(collectGraphicsLibrary, &paths);
    QStringList ordered = paths.values();
    ordered.sort();
    for (const QString &path : ordered) libraries.append(path);
    return libraries;
}

int main(int argc, char **argv) {
    qputenv("QT_QUICK_CONTROLS_STYLE", "Basic");
    qputenv("QSG_INFO", "1");
    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGLES);
    format.setRedBufferSize(8);
    format.setGreenBufferSize(8);
    format.setBlueBufferSize(8);
    format.setAlphaBufferSize(0);
    format.setDepthBufferSize(0);
    format.setStencilBufferSize(0);
    format.setSamples(0);
    format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    QSurfaceFormat::setDefaultFormat(format);
    QGuiApplication app(argc, argv);
    const QStringList args = app.arguments();
    const int duration = std::max(0, intArg(args, "--duration", 30));
    const double fps = std::max(1, intArg(args, "--fps", 60));
    const int outputIndex = args.indexOf("--output-json");
    const QString output = outputIndex >= 0 && outputIndex + 1 < args.size()
                                ? args.at(outputIndex + 1)
                                : QStringLiteral("/tmp/qt_gpu_probe.json");
    const QString manifestPath = stringArg(args, "--replay-manifest");
    const QStringList replayFrames = readReplayManifest(manifestPath);
    if (!manifestPath.isEmpty() && replayFrames.isEmpty()) return 4;
    const QString aiFrameFile = stringArg(args, "--ai-frame-file");
    const double aiHz = std::max(0.1, static_cast<double>(intArg(args, "--ai-hz", 10)));
    const QString pgvFile = stringArg(args, "--pgv-file");
    const QString pgvRing = stringArg(args, "--pgv-ring");
    const QString rawReplay = stringArg(args, "--raw-replay");
    const QString sourceLabel = stringArg(args, "--source-label");
    QString pciePgvFile = stringArg(args, "--pcie-pgv-file");
    if (pciePgvFile.isEmpty()) pciePgvFile = QStringLiteral("/dev/shm/pango-video/latest.pgv");
    QString localPgvFile = stringArg(args, "--local-pgv-file");
    if (localPgvFile.isEmpty()) localPgvFile = QStringLiteral("/dev/shm/pgv_local/latest.pgv");
    const bool autoCycleCapture = args.contains("--capture-cycle");
    const bool sceneEnabled = !args.contains("--scene-off");
    const QString pgvLayout = stringArg(args, "--pgv-layout");
    const int roadQuadrant = intArg(args, "--road-quadrant", 1);
    const int startMode = std::clamp(intArg(args, "--start-mode", 0), 0, 2);
    const QString effectiveLayout = pgvLayout.isEmpty()
                                        ? ((pgvFile.isEmpty() && rawReplay.isEmpty())
                                               ? QStringLiteral("front")
                                               : QStringLiteral("quad"))
                                        : pgvLayout;
    if (!pgvFile.isEmpty() && !QFileInfo::exists(pgvFile)) {
        qWarning().noquote() << "PGVF file does not exist yet:" << pgvFile;
    }
    if (!pgvRing.isEmpty() && !QFileInfo::exists(pgvRing)) {
        qWarning().noquote() << "PGV ring does not exist yet:" << pgvRing;
    }

    qmlRegisterType<VideoItem>("QtGpuProbe", 1, 0, "VideoItem");
    qmlRegisterType<TeslaSceneItem>("QtGpuProbe", 1, 0, "TeslaSceneItem");
    qmlRegisterType<HudControl>("QtGpuProbe", 1, 0, "HudControl");
    qmlRegisterType<PcieIspControl>("QtGpuProbe", 1, 0, "PcieIspControl");
    QQuickView view;
    view.setTitle(QStringLiteral("Qt Quick GPU Probe - RK3566"));
    view.setResizeMode(QQuickView::SizeRootObjectToView);
    view.rootContext()->setContextProperty(QStringLiteral("pgvInputLayout"), effectiveLayout);
    view.rootContext()->setContextProperty(QStringLiteral("pgvRoadQuadrant"), roadQuadrant);
    view.rootContext()->setContextProperty(QStringLiteral("initialViewMode"), startMode);
    view.rootContext()->setContextProperty(QStringLiteral("autoCycleCapture"), autoCycleCapture);
    view.rootContext()->setContextProperty(QStringLiteral("sceneEnabled"), sceneEnabled);
    view.rootContext()->setContextProperty(QStringLiteral("pciePgvFile"), pciePgvFile);
    view.rootContext()->setContextProperty(QStringLiteral("localPgvFile"), localPgvFile);
    const bool protocolPanelTest = qEnvironmentVariableIsSet("QT_PROTOCOL_PANEL_TEST");
    view.rootContext()->setContextProperty(QStringLiteral("protocolPanelTest"), protocolPanelTest);
    auto *ispControl = new PcieIspControl(&app);
    g_ispControl = ispControl;
    view.rootContext()->setContextProperty(QStringLiteral("fpgaIsp"), ispControl);
    qmlRegisterSingletonType<PcieIspControl>("QtGpuProbe", 1, 0, "PcieControl",
        &ispControlSingleton);
    const QString qmlOverride = qEnvironmentVariable("QT_QML_FILE");
    view.setSource(qmlOverride.isEmpty() ? QUrl(QStringLiteral("qrc:/main.qml"))
                                         : QUrl::fromLocalFile(qmlOverride));
    if (auto *rootObject = view.rootObject()) {
        rootObject->setProperty("fpgaControl", QVariant::fromValue(static_cast<QObject *>(ispControl)));
    }
    auto *video = view.rootObject() ? view.rootObject()->findChild<VideoItem *>(QStringLiteral("videoProbeItem")) : nullptr;
    auto *navBridge = new NavBridge(&app);
    QObject::connect(video, &VideoItem::aiChanged, navBridge, [video, navBridge] {
        navBridge->updateNav(video->aiNavigationDirection(),
                             video->aiNavigationDistanceM());
    });
    if (!video) {
        qCritical() << "QML VideoItem was not created";
        return 2;
    }
    QObject::connect(&view, &QQuickWindow::sceneGraphError, &app,
                     [&](QQuickWindow::SceneGraphError error, const QString &message) {
                         qCritical().noquote() << "SCENEGRAPH_ERROR" << error << message;
                         app.exit(3);
                     });
    QElapsedTimer runClock;
    runClock.start();
    view.showFullScreen();
    if (!rawReplay.isEmpty()) {
        video->startRawReplay(fps, rawReplay, aiFrameFile, aiHz);
    } else if (!pgvRing.isEmpty()) {
        video->startPgvRingProducer(fps, pgvRing, aiFrameFile, aiHz,
                                    sourceLabel.isEmpty() ? QStringLiteral("pgv-local") : sourceLabel);
    } else if (!pgvFile.isEmpty()) {
        video->startPgvProducer(fps, pgvFile, aiFrameFile, aiHz,
                                sourceLabel.isEmpty() ? QStringLiteral("pgv-live") : sourceLabel);
    } else {
        video->startProducer(fps, replayFrames, aiFrameFile, aiHz);
    }

    QTimer stopTimer;
    if (duration > 0) {
        stopTimer.setSingleShot(true);
        QObject::connect(&stopTimer, &QTimer::timeout, &app, &QCoreApplication::quit);
        stopTimer.start(duration * 1000);
    }
    const int result = app.exec();
    video->stopProducer();

    QJsonObject report;
    report["board_clock_note"] = "board system clock is not synchronized; see README run date";
    report["requested_input_fps"] = fps;
    const double actualDuration = std::max(0.001, runClock.elapsed() / 1000.0);
    report["duration_s"] = actualDuration;
    report["source_width"] = 1280;
    report["source_height"] = 720;
    report["source_format"] = "RGB565_LE";
    report["frame_source"] = video->sourceMode();
    report["pgv_file"] = pgvFile;
    report["pgv_ring"] = pgvRing;
    report["raw_replay"] = rawReplay;
    report["pgv_layout"] = effectiveLayout;
    report["pgv_road_quadrant"] = roadQuadrant;
    report["start_mode"] = startMode;
    report["auto_cycle_capture"] = autoCycleCapture;
    report["source_online"] = video->sourceOnline();
    report["source_fps"] = video->sourceFps();
    report["source_frame_id"] = static_cast<qint64>(video->sourceFrameId());
    report["source_dropped_frames"] = static_cast<qint64>(video->sourceDroppedFrames());
    report["source_crc_errors"] = static_cast<qint64>(video->sourceCrcErrors());
    report["npu_load_percent_last"] = video->npuLoadPercent();
    report["replay_manifest"] = manifestPath;
    report["replay_manifest_frame_count"] = replayFrames.size();
    report["input_fps_last_window"] = video->inputFps();
    report["consumed_fps_last_window"] = video->consumedFps();
    report["input_fps_average"] = video->producedFrames() / actualDuration;
    report["unique_source_fps_average"] = video->consumedFrames() / actualDuration;
    report["unique_frame_id_upload_fps_average"] = video->consumedFrames() / actualDuration;
    report["unique_replay_content_frames_uploaded"] =
        static_cast<qint64>(video->uniqueReplayContentFramesUploaded());
    report["unique_replay_content_fps_average"] =
        replayFrames.isEmpty() ? video->consumedFrames() / actualDuration
                               : video->uniqueReplayContentFramesUploaded() / actualDuration;
    const bool replayContentWrapped = !replayFrames.isEmpty() &&
        video->producedFrames() > static_cast<qulonglong>(replayFrames.size());
    report["replay_content_wrapped"] = replayContentWrapped;
    report["replay_content_note"] = replayFrames.isEmpty()
        ? "generated pattern changes with each producer frame ID"
        : replayContentWrapped
            ? "producer frame IDs advance at requested FPS; source image indexes wrapped past the replay manifest"
            : "source image indexes stayed within the replay manifest without wrapping in this run";
    report["qt_scene_graph_render_callback_fps_last_window"] = video->renderFps();
    const qint64 uploadedFrames = static_cast<qint64>(video->consumedFrames());
    const qint64 producedFrames = static_cast<qint64>(video->producedFrames());
    report["produced_frames"] = producedFrames;
    report["consumed_frames"] = uploadedFrames;
    report["successfully_uploaded_frames"] = uploadedFrames;
    report["source_frames_not_uploaded"] = std::max<qint64>(0, producedFrames - uploadedFrames);
    report["texture_upload_cpu_call_ms_last"] = video->uploadMs();
    report["texture_upload_p95_ms"] = video->uploadP95Ms();
    report["source_to_upload_ms_last"] = video->sourceToUploadMs();
    report["source_to_upload_p50_ms"] = video->sourceToUploadP50Ms();
    report["source_to_upload_p95_ms"] = video->sourceToUploadP95Ms();
    report["display_queue_dropped_on_take"] = static_cast<qint64>(video->queueDroppedOnTake());
    report["display_queue_overwritten"] = static_cast<qint64>(video->queueOverwritten());
    report["display_queue_max_depth"] = static_cast<qint64>(video->queueMaxDepth());
    report["display_queue_last_take_depth"] = static_cast<qint64>(video->queueLastTakeDepth());
    report["display_queue_taken"] = static_cast<qint64>(video->queueTaken());
    report["frame_push_total"] = static_cast<qint64>(video->queuePushed());
    report["frame_update_request_total"] = static_cast<qint64>(video->updateRequests());
    report["frame_update_handled_total"] = static_cast<qint64>(video->updateHandled());
    report["frame_take_total"] = static_cast<qint64>(video->queueTaken());
    report["frame_upload_total"] = static_cast<qint64>(video->consumedFrames());
    report["frame_drop_total"] = static_cast<qint64>(video->queueDroppedOnTake() + video->queueOverwritten());
    report["render_without_new_frame_total"] = static_cast<qint64>(video->renderWithoutNew());
    report["render_same_id_total"] = static_cast<qint64>(video->renderSameId());
    report["take_empty_total"] = static_cast<qint64>(video->takeEmpty());
    if (video->bboxModel()) {
        report["bbox_model_resets"] = video->bboxModel()->resetCount();
        report["bbox_model_inserts"] = video->bboxModel()->insertCount();
        report["bbox_model_removes"] = video->bboxModel()->removeCount();
        report["bbox_model_data_changed"] = video->bboxModel()->dataChangedCount();
    }
    report["process_cpu_percent_of_all_cores_last_window"] = video->cpuPercent();
    report["process_rss_mb_last"] = video->rssMb();
    report["mali_gpu_load_percent_last"] = video->gpuLoadPercent();
    report["mali_gpu_frequency_hz_last"] = static_cast<qint64>(video->gpuFrequencyHz());
    report["last_frame_id"] = static_cast<qint64>(video->frameId());
    report["gl_vendor"] = video->vendor();
    report["gl_renderer"] = video->renderer();
    report["gl_version"] = video->glVersion();
    report["egl_vendor"] = video->eglVendor();
    report["loaded_graphics_libraries"] = loadedGraphicsLibraries();
    report["hardware_accelerated"] = video->hardwareAccelerated();
    report["ai_requests_sent"] = static_cast<qint64>(video->aiRequestsSent());
    report["ai_results_received"] = static_cast<qint64>(video->aiResultsReceived());
    report["ai_online"] = video->aiOnline();
    report["ai_fps"] = video->aiFps();
    report["ai_latest_result_frame_id"] = static_cast<qint64>(video->latestAiFrameId());
    report["ai_detection_count_last"] = video->aiDetectionCount();
    report["ai_status_last"] = video->aiStatus();
    report["ai_traffic_last"] = video->aiTraffic();
    report["ai_navigation_last"] = video->aiNavigation();
    report["ai_navigation_road_last"] = video->aiNavigationRoad();
    report["ai_navigation_state_last"] = video->aiNavigationState();
    report["ai_navigation_direction_last"] = video->aiNavigationDirection();
    report["ai_navigation_distance_m_last"] = video->aiNavigationDistanceM();
    report["ai_result_age_ms_last"] = video->aiResultAgeMs();
    QJsonObject countObject;
    const QVariantMap counts = video->aiDetectionCounts();
    for (auto it = counts.begin(); it != counts.end(); ++it) {
        countObject.insert(it.key(), it.value().toInt());
    }
    report["ai_detection_counts_last"] = countObject;
    report["physical_scanout_fps"] = QJsonValue::Null;
    QFile file(output);
    if (file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        file.write(QJsonDocument(report).toJson(QJsonDocument::Indented));
        file.close();
    } else {
        qWarning().noquote() << "Could not write report" << output;
    }
    return result;
}
