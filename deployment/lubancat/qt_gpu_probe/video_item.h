#pragma once

#include <QQuickFramebufferObject>
#include <QMutex>
#include <QSet>
#include <QStringList>
#include <QVariantList>
#include <QTimer>
#include <QElapsedTimer>
#include <atomic>
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
    Q_PROPERTY(QString aiTraffic READ aiTraffic NOTIFY metricsChanged)
    Q_PROPERTY(QString aiNavigation READ aiNavigation NOTIFY metricsChanged)
    Q_PROPERTY(int aiDetectionCount READ aiDetectionCount NOTIFY metricsChanged)
    Q_PROPERTY(QVariantList aiBoxes READ aiBoxes NOTIFY metricsChanged)
    Q_PROPERTY(QVariantList aiHazards READ aiHazards NOTIFY metricsChanged)
    Q_PROPERTY(qulonglong aiRequestsSent READ aiRequestsSent NOTIFY metricsChanged)
    Q_PROPERTY(qulonglong aiResultsReceived READ aiResultsReceived NOTIFY metricsChanged)

public:
    struct Frame {
        uint64_t id = 0;
        std::vector<unsigned char> pixels;
        qint64 producedNs = 0;
        uint64_t sourceIndex = 0;
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
    QString aiTraffic() const;
    QString aiNavigation() const;
    int aiDetectionCount() const;
    QVariantList aiBoxes() const;
    QVariantList aiHazards() const;
    qulonglong aiRequestsSent() const { return ai_requests_sent_.load(); }
    qulonglong aiResultsReceived() const { return ai_results_received_.load(); }
    qulonglong uniqueReplayContentFramesUploaded() const {
        return render_metrics_->unique_replay_content_frames.load();
    }
    qulonglong latestAiFrameId() const { return latest_ai_frame_id_.load(); }

    void startProducer(double fps, const QStringList &replayFrames = {},
                       const QString &aiFrameFile = QString(), double aiHz = 10.0);
    void stopProducer();
    void updateMetrics();
    std::shared_ptr<Frame> takeLatest();

public slots:
    void requestFrameUpdate();

signals:
    void metricsChanged();

private:
    void produceFrame(uint64_t id);
    void runAiBridge(double fps, double aiHz, QString frameFile);

    mutable QMutex frame_mutex_;
    std::shared_ptr<Frame> latest_frame_;
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
    mutable QMutex ai_metrics_mutex_;
    QString ai_status_ = QStringLiteral("AI sidecar 未连接");
    QString ai_traffic_ = QStringLiteral("未知");
    QString ai_navigation_ = QStringLiteral("等待导航识别");
    QVariantList ai_boxes_;
    QVariantList ai_hazards_;
    std::atomic<uint64_t> latest_ai_frame_id_{0};
};
