#pragma once

#include <QQuickFramebufferObject>
#include <deque>
#include <fstream>
#include <vector>
#include <QMutex>
#include <QRectF>
#include <QSet>
#include <QStringList>
#include <QVariantList>
#include <QVariantMap>
#include <QTimer>
#include <QElapsedTimer>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class VideoItem : public QQuickFramebufferObject {
    Q_OBJECT
    Q_PROPERTY(double inputFps READ inputFps NOTIFY metricsChanged)
    Q_PROPERTY(double consumedFps READ consumedFps NOTIFY metricsChanged)
    Q_PROPERTY(double uploadMs READ uploadMs NOTIFY metricsChanged)
    Q_PROPERTY(double uploadP95Ms READ uploadP95Ms NOTIFY metricsChanged)
    Q_PROPERTY(double sourceToUploadMs READ sourceToUploadMs NOTIFY metricsChanged)
    Q_PROPERTY(double renderFps READ renderFps NOTIFY metricsChanged)
    Q_PROPERTY(double cpuPercent READ cpuPercent NOTIFY metricsChanged)
    Q_PROPERTY(double rssMb READ rssMb NOTIFY metricsChanged)
    Q_PROPERTY(double gpuLoadPercent READ gpuLoadPercent NOTIFY metricsChanged)
    Q_PROPERTY(qulonglong gpuFrequencyHz READ gpuFrequencyHz NOTIFY metricsChanged)
    Q_PROPERTY(qulonglong frameId READ frameId NOTIFY metricsChanged)
    Q_PROPERTY(qulonglong producedFrames READ producedFrames NOTIFY metricsChanged)
    Q_PROPERTY(qulonglong consumedFrames READ consumedFrames NOTIFY metricsChanged)
    Q_PROPERTY(qulonglong droppedFrames READ droppedFrames NOTIFY metricsChanged)
    Q_PROPERTY(QString renderer READ renderer NOTIFY metricsChanged)
    Q_PROPERTY(QString vendor READ vendor NOTIFY metricsChanged)
    Q_PROPERTY(QString glVersion READ glVersion NOTIFY metricsChanged)
    Q_PROPERTY(QString eglVendor READ eglVendor NOTIFY metricsChanged)
    Q_PROPERTY(bool hardwareAccelerated READ hardwareAccelerated NOTIFY metricsChanged)
    Q_PROPERTY(QString aiStatus READ aiStatus NOTIFY metricsChanged)
    Q_PROPERTY(bool aiOnline READ aiOnline NOTIFY metricsChanged)
    Q_PROPERTY(double aiFps READ aiFps NOTIFY metricsChanged)
    Q_PROPERTY(QString aiTraffic READ aiTraffic NOTIFY aiChanged)
    Q_PROPERTY(QString aiNavigation READ aiNavigation NOTIFY aiChanged)
    Q_PROPERTY(QString aiNavigationRoad READ aiNavigationRoad NOTIFY aiChanged)
    Q_PROPERTY(QString aiNavigationState READ aiNavigationState NOTIFY aiChanged)
    Q_PROPERTY(QString aiNavigationDirection READ aiNavigationDirection NOTIFY aiChanged)
    Q_PROPERTY(double aiNavigationDistanceM READ aiNavigationDistanceM NOTIFY aiChanged)
    Q_PROPERTY(int aiResultAgeMs READ aiResultAgeMs NOTIFY metricsChanged)
    Q_PROPERTY(int aiHazardAgeMs READ aiHazardAgeMs NOTIFY metricsChanged)
    Q_PROPERTY(int aiDetectionCount READ aiDetectionCount NOTIFY aiChanged)
    Q_PROPERTY(QVariantMap aiDetectionCounts READ aiDetectionCounts NOTIFY aiChanged)
    Q_PROPERTY(QVariantList aiBoxes READ aiBoxes NOTIFY aiChanged)
    Q_PROPERTY(QVariantList aiHazards READ aiHazards NOTIFY aiChanged)
    Q_PROPERTY(qulonglong aiRequestsSent READ aiRequestsSent NOTIFY metricsChanged)
    Q_PROPERTY(qulonglong aiResultsReceived READ aiResultsReceived NOTIFY metricsChanged)
    Q_PROPERTY(QString sourceMode READ sourceMode NOTIFY metricsChanged)
    Q_PROPERTY(bool sourceOnline READ sourceOnline NOTIFY metricsChanged)
    Q_PROPERTY(double sourceAgeMs READ sourceAgeMs NOTIFY metricsChanged)
    Q_PROPERTY(double sourceFps READ sourceFps NOTIFY metricsChanged)
    Q_PROPERTY(qulonglong sourceFrameId READ sourceFrameId NOTIFY metricsChanged)
    Q_PROPERTY(qulonglong displayFrameId READ displayFrameId NOTIFY frameAdvanced)
    Q_PROPERTY(qulonglong displayTimestampNs READ displayTimestampNs NOTIFY frameAdvanced)
    Q_PROPERTY(int aiFrameLag READ aiFrameLag NOTIFY aiChanged)
    Q_PROPERTY(int maxTrackPredictionMs READ maxTrackPredictionMs
               WRITE setMaxTrackPredictionMs NOTIFY metricsChanged)
    Q_PROPERTY(bool spatialViewActive READ spatialViewActive
               WRITE setSpatialViewActive NOTIFY spatialViewActiveChanged)
    Q_PROPERTY(qulonglong sourceDroppedFrames READ sourceDroppedFrames NOTIFY metricsChanged)
    Q_PROPERTY(qulonglong sourceCrcErrors READ sourceCrcErrors NOTIFY metricsChanged)
    Q_PROPERTY(double npuLoadPercent READ npuLoadPercent NOTIFY metricsChanged)
    Q_PROPERTY(QRectF sourceViewport READ sourceViewport WRITE setSourceViewport
               NOTIFY sourceViewportChanged)
    Q_PROPERTY(int aiRoadQuadrant READ aiRoadQuadrant WRITE setAiRoadQuadrant
               NOTIFY aiSelectionChanged)
    Q_PROPERTY(int aiNavigationQuadrant READ aiNavigationQuadrant WRITE setAiNavigationQuadrant
               NOTIFY aiSelectionChanged)

public:
    struct Frame {
        uint64_t id = 0;
        std::vector<unsigned char> pixels;
        qint64 producedNs = 0;
        qint64 sourceTimestampNs = 0;
        uint64_t sourceIndex = 0;
    };

    // Shared by the producer thread and the framebuffer renderer. Keeping the
    // queue independent of VideoItem lets the renderer survive scene-graph
    // teardown without dereferencing a deleted QQuickItem.
    class DisplayQueue {
    public:
        void push(const std::shared_ptr<Frame> &frame);
        std::shared_ptr<Frame> take();
        void clear();
        qulonglong droppedOnTake() const { return dropped_on_take_.load(); }
        qulonglong overwritten() const { return overwritten_.load(); }
        qulonglong maxDepth() const { return max_depth_.load(); }
        qulonglong lastTakeDepth() const { return last_take_depth_.load(); }
        qulonglong taken() const { return taken_.load(); }

    private:
        mutable QMutex mutex_;
        std::deque<std::shared_ptr<Frame>> frames_;
        std::atomic<qulonglong> dropped_on_take_{0};
        std::atomic<qulonglong> overwritten_{0};
        std::atomic<qulonglong> max_depth_{0};
        std::atomic<qulonglong> last_take_depth_{0};
        std::atomic<qulonglong> taken_{0};
    };

    struct RenderMetrics {
        std::atomic<uint64_t> uploaded_frames{0};
        std::atomic<uint64_t> frame_id{0};
        std::atomic<uint64_t> rendered_frames{0};
        std::atomic<double> upload_ms{0.0};
        std::atomic<double> source_to_upload_ms{0.0};
        std::atomic<double> render_fps{0.0};
        std::atomic<bool> hardware_accelerated{false};
        mutable QMutex strings_mutex;
        QString renderer;
        QString vendor;
        QString gl_version;
        QString egl_vendor;
        mutable QMutex upload_mutex;
        std::vector<double> upload_samples;
        mutable QMutex age_mutex;
        std::vector<double> source_age_samples;
        mutable QMutex replay_content_mutex;
        QSet<quint64> replay_content_indices;
        std::atomic<uint64_t> unique_replay_content_frames{0};
    };

    class Renderer;

    VideoItem(QQuickItem *parent = nullptr);
    ~VideoItem() override;

    QQuickFramebufferObject::Renderer *createRenderer() const override;

    double inputFps() const { return input_fps_.load(); }
    double consumedFps() const { return consumed_fps_.load(); }
    double uploadMs() const { return render_metrics_->upload_ms.load(); }
    double uploadP95Ms() const;
    double sourceToUploadMs() const { return render_metrics_->source_to_upload_ms.load(); }
    double sourceToUploadP50Ms() const;
    double sourceToUploadP95Ms() const;
    qulonglong queueDroppedOnTake() const { return display_queue_->droppedOnTake(); }
    qulonglong queueOverwritten() const { return display_queue_->overwritten(); }
    qulonglong queueMaxDepth() const { return display_queue_->maxDepth(); }
    qulonglong queueLastTakeDepth() const { return display_queue_->lastTakeDepth(); }
    qulonglong queueTaken() const { return display_queue_->taken(); }
    double renderFps() const { return render_metrics_->render_fps.load(); }
    double cpuPercent() const { return cpu_percent_.load(); }
    double rssMb() const { return rss_mb_.load(); }
    double gpuLoadPercent() const { return gpu_load_percent_.load(); }
    qulonglong gpuFrequencyHz() const { return static_cast<qulonglong>(gpu_frequency_hz_.load()); }
    qulonglong frameId() const { return static_cast<qulonglong>(render_metrics_->frame_id.load()); }
    qulonglong producedFrames() const { return static_cast<qulonglong>(produced_frames_.load()); }
    qulonglong consumedFrames() const { return static_cast<qulonglong>(render_metrics_->uploaded_frames.load()); }
    qulonglong droppedFrames() const {
        const uint64_t produced = produced_frames_.load();
        const uint64_t consumed = render_metrics_->uploaded_frames.load();
        return static_cast<qulonglong>(produced > consumed ? produced - consumed : 0);
    }
    QString renderer() const;
    QString vendor() const;
    QString glVersion() const;
    QString eglVendor() const;
    bool hardwareAccelerated() const { return render_metrics_->hardware_accelerated.load(); }
    QString aiStatus() const;
    bool aiOnline() const { return ai_online_.load(); }
    double aiFps() const { return ai_fps_.load(); }
    QString aiTraffic() const;
    QString aiNavigation() const;
    QString aiNavigationRoad() const;
    QString aiNavigationState() const;
    QString aiNavigationDirection() const;
    double aiNavigationDistanceM() const;
    int aiResultAgeMs() const { return ai_result_age_ms_.load(); }
    int aiHazardAgeMs() const { return ai_hazard_age_ms_.load(); }
    int aiDetectionCount() const;
    QVariantMap aiDetectionCounts() const;
    QVariantList aiBoxes() const;
    QVariantList aiHazards() const;
    qulonglong aiRequestsSent() const { return ai_requests_sent_.load(); }
    qulonglong aiResultsReceived() const { return ai_results_received_.load(); }
    QString sourceMode() const { return source_mode_; }
    bool pgvSource() const { return pgv_source_; }
    bool sourceOnline() const { return source_online_.load(); }
    double sourceAgeMs() const;
    double sourceFps() const { return source_fps_.load(); }
    qulonglong sourceFrameId() const { return source_frame_id_.load(); }
    qulonglong displayFrameId() const {
        const uint64_t rendered = rendered_frame_id_.load();
        return rendered ? rendered : source_frame_id_.load();
    }
    qulonglong displayTimestampNs() const {
        const qint64 rendered = rendered_timestamp_ns_.load();
        return rendered > 0 ? static_cast<qulonglong>(rendered) : 0;
    }
    int aiFrameLag() const { return ai_frame_lag_.load(); }
    int maxTrackPredictionMs() const { return max_track_prediction_ms_.load(); }
    void setMaxTrackPredictionMs(int value) {
        max_track_prediction_ms_.store(std::clamp(value, 50, 1000));
    }
    bool spatialViewActive() const { return spatial_view_active_.load(); }
    void setSpatialViewActive(bool value);
    void noteRenderedFrame(qulonglong id, qint64 timestamp_ns) {
        if (id) rendered_frame_id_.store(id);
        if (timestamp_ns > 0) rendered_timestamp_ns_.store(timestamp_ns);
    }
    qulonglong sourceDroppedFrames() const { return source_dropped_frames_.load(); }
    qulonglong sourceCrcErrors() const { return source_crc_errors_.load(); }
    double npuLoadPercent() const { return npu_load_percent_.load(); }
    QRectF sourceViewport() const;
    void setSourceViewport(const QRectF &viewport);
    int aiRoadQuadrant() const { return ai_road_quadrant_.load(); }
    int aiNavigationQuadrant() const { return ai_navigation_quadrant_.load(); }
    void setAiRoadQuadrant(int quadrant);
    void setAiNavigationQuadrant(int quadrant);
    qulonglong uniqueReplayContentFramesUploaded() const {
        return render_metrics_->unique_replay_content_frames.load();
    }
    qulonglong latestAiFrameId() const { return latest_ai_frame_id_.load(); }

    void startProducer(double fps, const QStringList &replayFrames = {},
                       const QString &aiFrameFile = QString(), double aiHz = 10.0);
    void startRawReplay(double fps, const QString &rawFile,
                        const QString &aiFrameFile = QString(), double aiHz = 10.0);
    void startPgvProducer(double fps, const QString &pgvFile,
                          const QString &aiFrameFile = QString(),
                          double aiHz = 10.0,
                          const QString &sourceLabel = QStringLiteral("pgv-live"));
    void stopProducer();
    void updateMetrics();

public slots:
    void requestFrameUpdate();

signals:
    void metricsChanged();
    void aiChanged();
    void frameAdvanced();
    void spatialViewActiveChanged();
    void sourceViewportChanged();
    void aiSelectionChanged();

private:
    void produceFrame(uint64_t id);
    void runPgvProducer(double fps, QString pgvFile, double aiHz, QString aiFrameFile);
    void runAiBridge(double fps, double aiHz, QString frameFile);
    void writeTeslaSceneTracks(const QVariantList &boxes, uint64_t frameId, qint64 timestampNs);

    std::shared_ptr<DisplayQueue> display_queue_;
    std::mutex ai_frame_mutex_;
    std::shared_ptr<Frame> latest_ai_frame_;
    QStringList replay_frames_;
    std::atomic<uint64_t> produced_frames_{0};
    std::atomic<double> input_fps_{0.0};
    std::atomic<double> consumed_fps_{0.0};
    std::atomic<double> cpu_percent_{0.0};
    std::atomic<double> rss_mb_{0.0};
    std::atomic<double> gpu_load_percent_{0.0};
    std::atomic<uint64_t> gpu_frequency_hz_{0};
    std::atomic<bool> producer_running_{false};
    std::atomic<bool> update_queued_{false};
    std::shared_ptr<RenderMetrics> render_metrics_ = std::make_shared<RenderMetrics>();
    QTimer metrics_timer_;
    QElapsedTimer stats_clock_;
    uint64_t last_produced_for_stats_ = 0;
    uint64_t last_consumed_for_stats_ = 0;
    uint64_t last_rendered_for_stats_ = 0;
    uint64_t last_cpu_ticks_ = 0;
    std::thread producer_thread_;
    std::thread ai_thread_;
    std::atomic<uint64_t> ai_requests_sent_{0};
    std::atomic<uint64_t> ai_results_received_{0};
    std::atomic<bool> ai_running_{false};
    std::atomic<uint32_t> ai_epoch_{0};
    uint32_t last_ai_epoch_ = 0;
    std::atomic<bool> ai_online_{false};
    std::atomic<bool> ai_dirty_{false};
    std::atomic<double> ai_fps_{0.0};
    std::atomic<int> ai_result_age_ms_{0};
    std::atomic<int> ai_hazard_age_ms_{0};
    std::atomic<qint64> last_ai_result_ns_{0};
    mutable QMutex ai_metrics_mutex_;
    QString ai_status_ = QStringLiteral("AI sidecar 未连接");
    QString ai_traffic_ = QStringLiteral("UNKNOWN");
    QString ai_navigation_ = QStringLiteral("等待导航识别");
    QString ai_navigation_road_;
    QString ai_navigation_state_ = QStringLiteral("UNKNOWN");
    QString ai_navigation_direction_ = QStringLiteral("none");
    double ai_navigation_distance_m_ = -1.0;
    QVariantMap ai_detection_counts_;
    QVariantList ai_boxes_;
    QVariantList ai_hazards_;
    std::atomic<uint64_t> latest_ai_frame_id_{0};
    std::atomic<int> ai_frame_lag_{0};
    std::atomic<int> max_track_prediction_ms_{250};
    std::atomic<bool> spatial_view_active_{false};
    std::atomic<uint64_t> rendered_frame_id_{0};
    std::atomic<qint64> rendered_timestamp_ns_{0};

    struct Track {
        uint64_t id = 0;
        int classId = 255;
        QString className;
        QString color;
        double x = 0.0;
        double y = 0.0;
        double width = 0.0;
        double height = 0.0;
        double vx = 0.0;
        double vy = 0.0;
        double confidence = 0.0;
        uint64_t lastDetectFrame = 0;
        uint64_t lastUpdateFrame = 0;
        qint64 lastDetectNs = 0;
        qint64 lastUpdateNs = 0;
        int missed = 0;
        int hits = 0;
        double vw = 0.0;
        double vh = 0.0;
        QString state = QStringLiteral("tentative");
        struct State {
            qint64 ns = 0;
            double x = 0.0;
            double y = 0.0;
            double width = 0.0;
            double height = 0.0;
        };
        std::deque<State> history;
    };
    std::vector<Track> tracks_;
    uint64_t next_track_id_ = 1;
    std::ofstream ai_timing_stream_;

    QString source_mode_ = QStringLiteral("generated");
    bool pgv_source_ = false;
    std::atomic<bool> source_online_{true};
    std::atomic<double> source_fps_{0.0};
    std::atomic<qint64> last_source_ns_{0};
    std::atomic<qulonglong> source_frame_id_{0};
    std::atomic<qulonglong> source_dropped_frames_{0};
    std::atomic<qulonglong> source_crc_errors_{0};
    std::atomic<double> npu_load_percent_{0.0};
    std::vector<std::shared_ptr<Frame>> pgv_pool_;
    mutable QMutex viewport_mutex_;
    QRectF source_viewport_{0.0, 0.0, 1.0, 1.0};
    std::atomic<int> ai_road_quadrant_{1};
    std::atomic<int> ai_navigation_quadrant_{1};
    void sendAiSelection(int road, int navigation);
};
