#include "video_item.h"

#include <QElapsedTimer>
#include <QColor>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QQuickWindow>
#include <QVariantMap>
#include <QtEndian>
#include <QtMath>
#include <EGL/egl.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <istream>
#include <QFile>
#include <QMetaObject>
#include <QThread>
#include <thread>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <cstdio>
#include <ctime>
#include <unistd.h>
#include <zlib.h>

namespace {
constexpr int kSourceWidth = 1280;
constexpr int kSourceHeight = 720;
constexpr size_t kFrameBytes = static_cast<size_t>(kSourceWidth) * kSourceHeight * 2;
constexpr size_t kPgvHeaderBytes = 40;
constexpr size_t kPgvFileBytes = kPgvHeaderBytes + kFrameBytes;

static qint64 steadyNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static uint32_t crc32Bytes(const unsigned char *bytes, size_t length) {
    // zlib's slice-by-8 CRC is materially cheaper than a byte-at-a-time table
    // on the RK3566 when validating 1.84 MB per frame at 60 Hz.
    uLong crc = ::crc32(0L, Z_NULL, 0);
    return static_cast<uint32_t>(::crc32(crc, bytes, static_cast<uInt>(length)));
}

static uint32_t readLe32(const unsigned char *bytes) {
    return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8U) |
           (static_cast<uint32_t>(bytes[2]) << 16U) | (static_cast<uint32_t>(bytes[3]) << 24U);
}

static uint64_t readLe64(const unsigned char *bytes) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= static_cast<uint64_t>(bytes[i]) << (8U * i);
    return value;
}

static QString classNameForId(uint32_t id) {
    switch (id) {
        case 0: return QStringLiteral("person");
        case 1: return QStringLiteral("bicycle");
        case 2: return QStringLiteral("car");
        case 3: return QStringLiteral("motorcycle");
        case 5: return QStringLiteral("bus");
        case 7: return QStringLiteral("truck");
        case 9: return QStringLiteral("traffic light");
        default: return QStringLiteral("object");
    }
}

static void readNpuCounter(double *load) {
    QFile file(QStringLiteral("/sys/class/devfreq/fde40000.npu/load"));
    if (!file.open(QIODevice::ReadOnly)) return;
    const QByteArray value = file.readAll().trimmed();
    const int at = value.indexOf('@');
    if (at > 0) *load = value.left(at).toDouble();
}

static uint64_t processCpuTicks() {
    QFile file(QStringLiteral("/proc/self/stat"));
    if (!file.open(QIODevice::ReadOnly)) return 0;
    const QByteArray line = file.readAll();
    const int close = line.lastIndexOf(')');
    if (close < 0) return 0;
    const auto fields = line.mid(close + 2).simplified().split(' ');
    if (fields.size() <= 12) return 0;
    return fields[11].toULongLong() + fields[12].toULongLong();
}
static double processRssMb() {
    QFile file(QStringLiteral("/proc/self/statm"));
    if (!file.open(QIODevice::ReadOnly)) return 0.0;
    const auto fields = file.readAll().simplified().split(' ');
    if (fields.size() < 2) return 0.0;
    bool ok = false;
    const qulonglong residentPages = fields.at(1).toULongLong(&ok);
    const long pageSize = sysconf(_SC_PAGESIZE);
    return ok && pageSize > 0 ? residentPages * static_cast<double>(pageSize) / (1024.0 * 1024.0) : 0.0;
}

static void readGpuCounters(double *load, uint64_t *frequency) {
    QFile file(QStringLiteral("/sys/devices/platform/fde60000.gpu/devfreq/fde60000.gpu/load"));
    if (!file.open(QIODevice::ReadOnly)) return;
    const QByteArray value = file.readAll().trimmed();
    const int at = value.indexOf('@');
    if (at > 0) *load = value.left(at).toDouble();
    if (at >= 0) {
        QByteArray hz = value.mid(at + 1);
        hz.replace("Hz", "");
        *frequency = hz.toULongLong();
    }
}

static uint16_t rgb565(int r, int g, int b) {
    return static_cast<uint16_t>(((r & 0xf8) << 8) | ((g & 0xfc) << 3) | (b >> 3));
}

static void fillRect(std::vector<unsigned char> &bytes, int x0, int y0, int x1, int y1,
                     uint16_t color) {
    x0 = std::max(0, std::min(kSourceWidth, x0));
    x1 = std::max(0, std::min(kSourceWidth, x1));
    y0 = std::max(0, std::min(kSourceHeight, y0));
    y1 = std::max(0, std::min(kSourceHeight, y1));
    for (int y = y0; y < y1; ++y) {
        auto *row = reinterpret_cast<uint16_t *>(bytes.data() + static_cast<size_t>(y) * kSourceWidth * 2);
        for (int x = x0; x < x1; ++x) row[x] = color;
    }
}

static void fillRoadFrame(std::vector<unsigned char> &bytes, uint64_t id) {
    bytes.resize(kFrameBytes);
    auto *pixels = reinterpret_cast<uint16_t *>(bytes.data());
    const int phase = static_cast<int>((id * 7) % 360);
    for (int y = 0; y < kSourceHeight; ++y) {
        const int horizon = 235;
        const int sky = y < horizon ? y : horizon;
        const int r = 8 + sky / 28;
        const int g = 18 + sky / 15;
        const int b = 34 + sky / 9;
        const uint16_t color = rgb565(r, g, b);
        for (int x = 0; x < kSourceWidth; ++x) pixels[y * kSourceWidth + x] = color;
    }
    fillRect(bytes, 0, 235, kSourceWidth, kSourceHeight, rgb565(24, 30, 35));
    for (int y = 235; y < kSourceHeight; ++y) {
        const int width = 42 + (y - 235) * 2;
        const int left = kSourceWidth / 2 - width;
        const int right = kSourceWidth / 2 + width;
        auto *row = reinterpret_cast<uint16_t *>(bytes.data() + static_cast<size_t>(y) * kSourceWidth * 2);
        for (int x = left; x < right; ++x) {
            if (x >= 0 && x < kSourceWidth) row[x] = rgb565(31 + (y % 9), 36 + (y % 7), 41 + (y % 5));
        }
        if (((y - 235 + phase) / 38) % 2 == 0) {
            const int lane = kSourceWidth / 2 - width / 2;
            for (int dx = -3; dx <= 3; ++dx) if (lane + dx >= 0 && lane + dx < kSourceWidth) row[lane + dx] = rgb565(220, 220, 188);
        }
    }
    // A moving cyan marker makes every source frame visually distinguishable.
    const int marker_x = 80 + static_cast<int>((id * 11) % 1020);
    const int marker_y = 75 + static_cast<int>((id * 3) % 80);
    fillRect(bytes, marker_x, marker_y, marker_x + 120, marker_y + 34, rgb565(0, 170, 210));
    fillRect(bytes, marker_x + 8, marker_y + 8, marker_x + 112, marker_y + 26, rgb565(8, 22, 36));
}
} // namespace

class VideoItem::Renderer final : public QQuickFramebufferObject::Renderer {
public:
    Renderer(std::shared_ptr<VideoItem::DisplayQueue> display,
             std::shared_ptr<VideoItem::RenderMetrics> metrics)
        : display_(std::move(display)), metrics_(std::move(metrics)) {}

    ~Renderer() override {
        // Scene-graph renderers are destroyed on the render thread. Release the
        // texture while its owning context is still current when possible.
        if (texture_) {
            if (auto *context = QOpenGLContext::currentContext()) {
                context->functions()->glDeleteTextures(1, &texture_);
            }
            texture_ = 0;
        }
    }

    QOpenGLFramebufferObject *createFramebufferObject(const QSize &size) override {
        QOpenGLFramebufferObjectFormat format;
        format.setAttachment(QOpenGLFramebufferObject::CombinedDepthStencil);
        format.setInternalTextureFormat(GL_RGBA);
        return new QOpenGLFramebufferObject(size, format);
    }

    void synchronize(QQuickFramebufferObject *item) override {
        auto *video = static_cast<VideoItem *>(item);
        viewport_ = video->sourceViewport();
        pgv_flip_ = video->pgvSource();
        metrics_->synchronize_calls.fetch_add(1);
        // Producer marks the item dirty for every accepted frame. Keep the
        // existing pending frame until render() uploads it; replacing it in
        // synchronize() was measured to add work without improving throughput.
        if (!pending_) pending_ = display_->take();
    }

    void render() override {
        metrics_->rendered_frames.fetch_add(1);
        auto *startContext = QOpenGLContext::currentContext();
        if (startContext) {
            GLenum startError = startContext->functions()->glGetError();
            if (startError != GL_NO_ERROR)
                qWarning() << "GL error present at VideoItem::render start" << startError;
        }
        if (!program_) initGL();
        if (!program_) return;
        auto *context = QOpenGLContext::currentContext();
        if (!context) return;
        auto *f = context->functions();
        const auto frame = pending_;
        if (frame && frame->id != uploaded_id_) {
            QElapsedTimer timer;
            timer.start();
            f->glActiveTexture(GL_TEXTURE0);
            GLenum stale = f->glGetError();
            while (stale != GL_NO_ERROR) {
                qWarning() << "pre-existing GL error before RGB565 upload" << stale;
                stale = f->glGetError();
            }
            GLenum uploadError = GL_NO_ERROR;
            if (!texture_) {
                f->glGenTextures(1, &texture_);
                if (!texture_) {
                    pending_.reset();
                    update();
                    return;
                }
                f->glBindTexture(GL_TEXTURE_2D, texture_);
                f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                f->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB565, kSourceWidth, kSourceHeight, 0,
                                GL_RGB, GL_UNSIGNED_SHORT_5_6_5, frame->pixels.data());
                uploadError = f->glGetError();
                if (uploadError != GL_NO_ERROR)
                    qWarning() << "RGB565 glTexImage2D failed" << uploadError;
            } else {
                f->glBindTexture(GL_TEXTURE_2D, texture_);
                f->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kSourceWidth, kSourceHeight,
                                   GL_RGB, GL_UNSIGNED_SHORT_5_6_5, frame->pixels.data());
                uploadError = f->glGetError();
                if (uploadError != GL_NO_ERROR)
                    qWarning() << "RGB565 glTexSubImage2D failed" << uploadError;
            }
            if (uploadError == GL_NO_ERROR) {
                uploaded_id_ = frame->id;
                const double uploadMs = static_cast<double>(timer.nsecsElapsed()) / 1.0e6;
                const qint64 nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                const qint64 sourceNs = frame->sourceTimestampNs > 0
                    ? frame->sourceTimestampNs : frame->producedNs;
                const double sourceAgeMs = std::max<qint64>(0, nowNs - sourceNs) / 1.0e6;
                metrics_->upload_ms.store(uploadMs);
                metrics_->source_to_upload_ms.store(sourceAgeMs);
                metrics_->frame_id.store(uploaded_id_);
                metrics_->uploaded_frames.fetch_add(1);
                {
                    QMutexLocker contentLocker(&metrics_->replay_content_mutex);
                    const quint64 sourceIndex = static_cast<quint64>(frame->sourceIndex);
                    if (!metrics_->replay_content_indices.contains(sourceIndex)) {
                        metrics_->replay_content_indices.insert(sourceIndex);
                        metrics_->unique_replay_content_frames.fetch_add(1);
                    }
                }
                QMutexLocker locker(&metrics_->upload_mutex);
                metrics_->upload_samples.push_back(uploadMs);
                if (metrics_->upload_samples.size() > 3600) {
                    metrics_->upload_samples.erase(metrics_->upload_samples.begin(),
                                                   metrics_->upload_samples.begin() + 600);
                }
                metrics_->source_age_samples.push_back(sourceAgeMs);
                if (metrics_->source_age_samples.size() > 3600) {
                    metrics_->source_age_samples.erase(metrics_->source_age_samples.begin(),
                                                       metrics_->source_age_samples.begin() + 600);
                }
            } else {
                qWarning() << "RGB565 texture upload failed with GL error" << uploadError;
            }
            pending_.reset();
        }
        f->glViewport(0, 0, framebufferObject()->width(), framebufferObject()->height());
        f->glDisable(GL_DEPTH_TEST);
        f->glClearColor(0.015f, 0.025f, 0.04f, 1.0f);
        f->glClear(GL_COLOR_BUFFER_BIT);
        program_->bind();
        f->glActiveTexture(GL_TEXTURE0);
        f->glBindTexture(GL_TEXTURE_2D, texture_);
        program_->setUniformValue("source", 0);
        const float u0 = static_cast<float>(viewport_.left());
        const float u1 = static_cast<float>(viewport_.right());
        // The QQuickFramebufferObject path composites a vertically mirrored
        // FBO on this X11/EGL board. Live PGVF rows need the normal top->bottom
        // V order; the older replay/manifest frames were captured with the
        // opposite orientation and retain the historical 1->0 mapping.
        const float vTop = pgv_flip_ ? static_cast<float>(viewport_.top())
                                     : static_cast<float>(1.0 - viewport_.top());
        const float vBottom = pgv_flip_ ? static_cast<float>(viewport_.bottom())
                                        : static_cast<float>(1.0 - viewport_.bottom());
        const GLfloat vertices[] = {
            -1.f, -1.f, u0, vBottom,  1.f, -1.f, u1, vBottom,
            -1.f,  1.f, u0, vTop,     1.f,  1.f, u1, vTop,
        };
        program_->enableAttributeArray(0);
        program_->setAttributeArray(0, GL_FLOAT, vertices, 2, 4 * sizeof(GLfloat));
        program_->enableAttributeArray(1);
        program_->setAttributeArray(1, GL_FLOAT, vertices + 2, 2, 4 * sizeof(GLfloat));
        f->glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        program_->disableAttributeArray(0);
        program_->disableAttributeArray(1);
        program_->release();
        {
            const GLenum endError = f->glGetError();
            if (endError != GL_NO_ERROR)
                qWarning() << "GL error at VideoItem::render end" << endError;
        }
        update();
    }

private:
    std::shared_ptr<VideoItem::DisplayQueue> display_;

    void initGL() {
        auto *context = QOpenGLContext::currentContext();
        if (!context) return;
        program_ = std::make_unique<QOpenGLShaderProgram>();
        const char *vertex = "attribute vec2 position; attribute vec2 texcoord; varying vec2 uv; void main(){ gl_Position=vec4(position,0.0,1.0); uv=texcoord; }";
        const char *fragment = "precision mediump float; varying vec2 uv; uniform sampler2D source; void main(){ gl_FragColor=texture2D(source,uv); }";
        if (!program_->addShaderFromSourceCode(QOpenGLShader::Vertex, vertex) ||
            !program_->addShaderFromSourceCode(QOpenGLShader::Fragment, fragment)) {
            program_.reset();
            return;
        }
        program_->bindAttributeLocation("position", 0);
        program_->bindAttributeLocation("texcoord", 1);
        if (!program_->link()) program_.reset();
        auto *f = context->functions();
        const auto *vendor = reinterpret_cast<const char *>(f->glGetString(GL_VENDOR));
        const auto *renderer = reinterpret_cast<const char *>(f->glGetString(GL_RENDERER));
        const auto *version = reinterpret_cast<const char *>(f->glGetString(GL_VERSION));
        const auto egl_vendor = eglQueryString(eglGetCurrentDisplay(), EGL_VENDOR);
        const QString vendor_q = QString::fromUtf8(vendor ? vendor : "unknown");
        const QString renderer_q = QString::fromUtf8(renderer ? renderer : "unknown");
        const QString version_q = QString::fromUtf8(version ? version : "unknown");
        const QString egl_q = QString::fromUtf8(egl_vendor ? egl_vendor : "unknown");
        const bool hardware = program_ &&
                        !renderer_q.contains("llvmpipe", Qt::CaseInsensitive) &&
                        !renderer_q.contains("softpipe", Qt::CaseInsensitive) &&
                        !renderer_q.contains("software", Qt::CaseInsensitive) &&
                        renderer_q.contains("Mali", Qt::CaseInsensitive);
        QMutexLocker locker(&metrics_->strings_mutex);
        metrics_->vendor = vendor_q;
        metrics_->renderer = renderer_q;
        metrics_->gl_version = version_q;
        metrics_->egl_vendor = egl_q;
        metrics_->hardware_accelerated.store(hardware);
    }

    std::shared_ptr<VideoItem::RenderMetrics> metrics_;
    std::shared_ptr<Frame> pending_;
    QRectF viewport_{0.0, 0.0, 1.0, 1.0};
    bool pgv_flip_ = false;
    uint64_t uploaded_id_ = UINT64_MAX;
    GLuint texture_ = 0;
    std::unique_ptr<QOpenGLShaderProgram> program_;
};

VideoItem::VideoItem(QQuickItem *parent) : QQuickFramebufferObject(parent) {
    display_queue_ = std::make_shared<DisplayQueue>();
    const QByteArray timingPath = qgetenv("QT_AI_TIMING_LOG");
    if (!timingPath.isEmpty()) {
        ai_timing_stream_.open(timingPath.constData(), std::ios::app);
    }
    // QQuickFramebufferObject composites its FBO with a vertically flipped
    // texture on the board's xcb/EGL path. Keep the source RGB565 rows
    // top-to-bottom and mirror the FBO once so road video is upright.
    setMirrorVertically(true);
    stats_clock_.start();
    connect(&metrics_timer_, &QTimer::timeout, this, &VideoItem::updateMetrics);
    metrics_timer_.start(200);
}

VideoItem::~VideoItem() { stopProducer(); }

QQuickFramebufferObject::Renderer *VideoItem::createRenderer() const {
    return new Renderer(display_queue_, render_metrics_);
}

void VideoItem::DisplayQueue::push(const std::shared_ptr<Frame> &frame) {
    QMutexLocker locker(&mutex_);
    if (frames_.size() >= 4) {
        frames_.pop_front();
        overwritten_.fetch_add(1);
    }
    frames_.push_back(frame);
    max_depth_.store(std::max<qulonglong>(max_depth_.load(), frames_.size()));
}

std::shared_ptr<VideoItem::Frame> VideoItem::DisplayQueue::take() {
    QMutexLocker locker(&mutex_);
    if (frames_.empty()) return {};
    const qulonglong depth = frames_.size();
    last_take_depth_.store(depth);
    taken_.fetch_add(1);
    // Latest-only display semantics: keep the newest complete frame and
    // discard older queued frames in the same critical section. AI has its
    // own latest_ai_frame_ slot and is not affected by this consumer.
    auto frame = std::move(frames_.back());
    dropped_on_take_.fetch_add(depth > 0 ? depth - 1 : 0);
    frames_.clear();
    return frame;
}

void VideoItem::DisplayQueue::clear() {
    QMutexLocker locker(&mutex_);
    frames_.clear();
}

void VideoItem::startProducer(double fps, const QStringList &replayFrames,
                              const QString &aiFrameFile, double aiHz) {
    stopProducer();
    display_queue_->clear();
    replay_frames_ = replayFrames;
    pgv_source_ = false;
    source_mode_ = replayFrames.isEmpty() ? QStringLiteral("generated")
                                          : QStringLiteral("replay");
    source_online_.store(true);
    last_source_ns_.store(steadyNs());
    const auto period = std::chrono::nanoseconds(
        static_cast<int64_t>(std::llround(1.0e9 / std::max(1.0, fps))));
    producer_running_.store(true);
    producer_thread_ = std::thread([this, period]() {
        auto next = std::chrono::steady_clock::now();
        while (producer_running_.load()) {
            const auto now = std::chrono::steady_clock::now();
            if (now < next) std::this_thread::sleep_until(next);
            if (!producer_running_.load()) break;
            const uint64_t id = produced_frames_.fetch_add(1) + 1;
            produceFrame(id);
            const auto after = std::chrono::steady_clock::now();
            next += period;
            if (next <= after) next = after + period;
        }
    });
    if (!aiFrameFile.isEmpty()) {
        ai_running_.store(true);
        ai_thread_ = std::thread(&VideoItem::runAiBridge, this, fps, aiHz, aiFrameFile);
    }
}

void VideoItem::startRawReplay(double fps, const QString &rawFile,
                               const QString &aiFrameFile, double aiHz) {
    stopProducer();
    display_queue_->clear();
    replay_frames_.clear();
    pgv_source_ = false;
    source_mode_ = QStringLiteral("local-raw");
    source_online_.store(true);
    last_source_ns_.store(steadyNs());
    const auto period = std::chrono::nanoseconds(
        static_cast<int64_t>(std::llround(1.0e9 / std::max(1.0, fps))));
    producer_running_.store(true);
    producer_thread_ = std::thread([this, period, path = rawFile.toStdString()]() {
        std::ifstream raw(path, std::ios::binary);
        if (!raw) {
            qWarning().noquote() << "Cannot open raw replay" << QString::fromStdString(path);
            producer_running_.store(false);
            return;
        }
        raw.seekg(0, std::ios::end);
        const auto size = static_cast<uint64_t>(raw.tellg());
        const uint64_t count = size / kFrameBytes;
        if (!count) {
            qWarning() << "Raw replay has no full 1280x720 RGB565 frames";
            producer_running_.store(false);
            return;
        }
        auto next = std::chrono::steady_clock::now();
        while (producer_running_.load()) {
            const auto now = std::chrono::steady_clock::now();
            if (now < next) std::this_thread::sleep_until(next);
            if (!producer_running_.load()) break;
            const uint64_t id = produced_frames_.fetch_add(1) + 1;
            auto frame = std::make_shared<Frame>();
            frame->pixels.resize(kFrameBytes);
            raw.seekg(static_cast<std::streamoff>(((id - 1) % count) * kFrameBytes));
            raw.read(reinterpret_cast<char *>(frame->pixels.data()),
                     static_cast<std::streamsize>(kFrameBytes));
            if (raw.gcount() != static_cast<std::streamsize>(kFrameBytes)) break;
            frame->id = id;
            frame->sourceIndex = (id - 1) % count;
            frame->producedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            source_frame_id_.store(frame->id);
            last_source_ns_.store(steadyNs());
            source_online_.store(true);
            display_queue_->push(frame);
            if (ai_running_.load()) {
                std::lock_guard<std::mutex> locker(ai_frame_mutex_);
                latest_ai_frame_ = std::move(frame);
            }
            if (!update_queued_.exchange(true) &&
                !QMetaObject::invokeMethod(this, "requestFrameUpdate", Qt::QueuedConnection)) {
                update_queued_.store(false);
            }
            const auto after = std::chrono::steady_clock::now();
            next += period;
            if (next <= after) next = after + period;
        }
    });
    if (!aiFrameFile.isEmpty()) {
        ai_running_.store(true);
        ai_thread_ = std::thread(&VideoItem::runAiBridge, this, fps, aiHz, aiFrameFile);
    }
}

void VideoItem::startPgvProducer(double fps, const QString &pgvFile,
                                 const QString &aiFrameFile, double aiHz,
                                 const QString &sourceLabel) {
    stopProducer();
    display_queue_->clear();
    replay_frames_.clear();
    pgv_source_ = true;
    source_mode_ = sourceLabel.isEmpty() ? QStringLiteral("pgv-live") : sourceLabel;
    source_online_.store(false);
    last_source_ns_.store(steadyNs());
    update();
    pgv_pool_.clear();
    for (int i = 0; i < 4; ++i) {
        auto frame = std::make_shared<Frame>();
        frame->pixels.resize(kFrameBytes);
        pgv_pool_.push_back(std::move(frame));
    }
    producer_running_.store(true);
    producer_thread_ = std::thread(&VideoItem::runPgvProducer, this, fps, pgvFile, aiHz, aiFrameFile);
    if (!aiFrameFile.isEmpty()) {
        ai_running_.store(true);
        ai_thread_ = std::thread(&VideoItem::runAiBridge, this, fps, aiHz, aiFrameFile);
    }
}

void VideoItem::stopProducer() {
    producer_running_.store(false);
    if (producer_thread_.joinable()) producer_thread_.join();
    ai_running_.store(false);
    if (ai_thread_.joinable()) ai_thread_.join();
}

void VideoItem::produceFrame(uint64_t id) {
    if (!producer_running_.load()) return;
    auto frame = std::make_shared<Frame>();
    frame->id = id;
    frame->sourceIndex = replay_frames_.isEmpty() ? id - 1 : (id - 1) % static_cast<uint64_t>(replay_frames_.size());
    frame->producedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    frame->sourceTimestampNs = frame->producedNs;
    if (replay_frames_.isEmpty()) {
        fillRoadFrame(frame->pixels, id);
    } else {
        QFile input(replay_frames_.at(static_cast<int>(frame->sourceIndex)));
        if (!input.open(QIODevice::ReadOnly)) {
            qWarning().noquote() << "Cannot read replay frame" << input.fileName() << input.errorString();
            producer_running_.store(false);
            return;
        }
        frame->pixels.resize(kFrameBytes);
        qint64 bytesRead = 0;
        while (bytesRead < static_cast<qint64>(kFrameBytes)) {
            const qint64 count = input.read(
                reinterpret_cast<char *>(frame->pixels.data()) + bytesRead,
                static_cast<qint64>(kFrameBytes) - bytesRead);
            if (count <= 0) break;
            bytesRead += count;
        }
        if (bytesRead != static_cast<qint64>(kFrameBytes)) {
            qWarning().noquote() << "Invalid RGB565 replay frame size" << input.fileName() << bytesRead;
            producer_running_.store(false);
            return;
        }
    }
    source_frame_id_.store(frame->id);
    last_source_ns_.store(steadyNs());
    source_online_.store(true);
    display_queue_->push(frame);
    if (ai_running_.load()) {
        std::lock_guard<std::mutex> locker(ai_frame_mutex_);
        latest_ai_frame_ = std::move(frame);
    }
    if (!update_queued_.exchange(true) &&
        !QMetaObject::invokeMethod(this, "requestFrameUpdate", Qt::QueuedConnection)) {
        update_queued_.store(false);
    }
}

void VideoItem::runPgvProducer(double fps, QString pgvFile, double aiHz, QString aiFrameFile) {
    (void)fps;
    (void)aiHz;
    (void)aiFrameFile;
    const std::string path = pgvFile.toStdString();
    // CRC is mandatory by default. The local PGVF publisher uses atomic
    // rename, so a faster local-only mode can skip CRC after header/length
    // validation; real PCIe keeps CRC unless explicitly opted out.
    const bool verifyCrc = qEnvironmentVariableIntValue("QT_PGVF_SKIP_CRC") == 0;
    std::array<unsigned char, kPgvHeaderBytes> header{};
    struct stat previous {};
    bool havePrevious = false;
    bool haveFrame = false;
    uint32_t lastGeneration = 0;
    uint64_t lastSourceFrame = 0;

    while (producer_running_.load()) {
        struct stat info {};
        if (::stat(path.c_str(), &info) != 0 || info.st_size != static_cast<off_t>(kPgvFileBytes)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            continue;
        }
        if (havePrevious && info.st_dev == previous.st_dev && info.st_ino == previous.st_ino &&
            info.st_mtim.tv_sec == previous.st_mtim.tv_sec &&
            info.st_mtim.tv_nsec == previous.st_mtim.tv_nsec && info.st_size == previous.st_size) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        std::shared_ptr<Frame> slot;
        for (const auto &candidate : pgv_pool_) {
            if (candidate.use_count() == 1) {
                slot = candidate;
                break;
            }
        }
        if (!slot) {
            previous = info;
            havePrevious = true;
            source_dropped_frames_.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            continue;
        }
        const ssize_t headerGot = ::pread(fd, header.data(), header.size(), 0);
        if (headerGot != static_cast<ssize_t>(header.size())) {
            ::close(fd);
            source_crc_errors_.fetch_add(1);
            previous = info;
            havePrevious = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        const uint32_t version = readLe32(header.data() + 4);
        const uint32_t width = readLe32(header.data() + 8);
        const uint32_t height = readLe32(header.data() + 12);
        const uint32_t sourceFrame = readLe32(header.data() + 16);
        const uint32_t generation = readLe32(header.data() + 20);
        const uint32_t expectedCrc = readLe32(header.data() + 24);
        const uint32_t payloadBytes = readLe32(header.data() + 28);
        const uint64_t timestampNs = readLe64(header.data() + 32);
        if (std::memcmp(header.data(), "PGVF", 4) != 0 || version != 1 ||
            width != kSourceWidth || height != kSourceHeight ||
            payloadBytes != kFrameBytes || timestampNs == 0) {
            source_crc_errors_.fetch_add(1);
            previous = info;
            havePrevious = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        const ssize_t payloadGot = ::pread(fd, slot->pixels.data(), kFrameBytes, kPgvHeaderBytes);
        ::close(fd);
        previous = info;
        havePrevious = true;
        if (payloadGot != static_cast<ssize_t>(kFrameBytes) ||
            (verifyCrc && crc32Bytes(slot->pixels.data(), kFrameBytes) != expectedCrc)) {
            source_crc_errors_.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        if (haveFrame && generation != lastGeneration) {
            // Local PGVF cycles increment generation. Start a new tracker epoch
            // so the tail of the previous loop cannot be associated with the
            // first vehicles of the next loop.
            ai_epoch_.fetch_add(1);
        } else if (haveFrame && generation == lastGeneration) {
            const uint32_t delta = sourceFrame - static_cast<uint32_t>(lastSourceFrame);
            if (delta == 0 || delta >= 0x80000000U) continue;
            source_dropped_frames_.fetch_add(delta - 1);
        }
        slot->id = sourceFrame;
        slot->sourceIndex = sourceFrame;
        slot->producedNs = steadyNs();
        slot->sourceTimestampNs = static_cast<qint64>(timestampNs);
        source_frame_id_.store(sourceFrame);
        last_source_ns_.store(slot->producedNs);
        source_online_.store(true);
        lastGeneration = generation;
        lastSourceFrame = sourceFrame;
        haveFrame = true;
        produced_frames_.fetch_add(1);
        display_queue_->push(slot);
        if (ai_running_.load()) {
            std::lock_guard<std::mutex> locker(ai_frame_mutex_);
            latest_ai_frame_ = slot;
        }
        // Mark the QQuickFramebufferObject dirty once per accepted frame. The
        // renderer takes the newest queued frame in synchronize() and uploads
        // it; without this event the compositor keeps reusing the old texture.
        if (!update_queued_.exchange(true) &&
            !QMetaObject::invokeMethod(this, "requestFrameUpdate", Qt::QueuedConnection)) {
            update_queued_.store(false);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void VideoItem::runAiBridge(double fps, double aiHz, QString frameFile) {
    const int socketFd = socket(AF_INET, SOCK_DGRAM, 0);
    if (socketFd < 0) {
        QMutexLocker locker(&ai_metrics_mutex_);
        ai_status_ = QStringLiteral("AI UDP socket 创建失败");
        return;
    }
    const int flags = fcntl(socketFd, F_GETFL, 0);
    if (flags >= 0) fcntl(socketFd, F_SETFL, flags | O_NONBLOCK);
    sockaddr_in resultAddress{};
    resultAddress.sin_family = AF_INET;
    resultAddress.sin_port = htons(38261);
    resultAddress.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(socketFd, reinterpret_cast<const sockaddr *>(&resultAddress), sizeof(resultAddress)) != 0) {
        close(socketFd);
        QMutexLocker locker(&ai_metrics_mutex_);
        ai_status_ = QStringLiteral("AI UDP 38261 绑定失败");
        return;
    }
    sockaddr_in requestAddress{};
    requestAddress.sin_family = AF_INET;
    requestAddress.sin_port = htons(38260);
    requestAddress.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const auto interval = std::chrono::nanoseconds(
        static_cast<int64_t>(std::llround(1.0e9 / std::max(0.1, aiHz))));
    auto nextRequest = std::chrono::steady_clock::now();
    uint64_t latestResultId = 0;
    std::array<unsigned char, 8192> packet{};
    const std::string outputPath = frameFile.toUtf8().toStdString();
    const std::string temporaryPath = outputPath + ".tmp";

    while (ai_running_.load()) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= nextRequest) {
            std::shared_ptr<Frame> frame;
            {
                std::lock_guard<std::mutex> locker(ai_frame_mutex_);
                frame = std::move(latest_ai_frame_);
            }
            if (frame && !frame->pixels.empty()) {
                QByteArray header(28, '\0');
                std::memcpy(header.data(), "MFR1", 4);
                qToLittleEndian<quint64>(frame->id, reinterpret_cast<uchar *>(header.data() + 4));
                qToLittleEndian<quint64>(static_cast<quint64>(frame->producedNs), reinterpret_cast<uchar *>(header.data() + 12));
                qToLittleEndian<quint32>(kSourceWidth * 2, reinterpret_cast<uchar *>(header.data() + 20));
                qToLittleEndian<quint32>(static_cast<quint32>(kFrameBytes), reinterpret_cast<uchar *>(header.data() + 24));
                std::ofstream snapshot(temporaryPath, std::ios::binary | std::ios::trunc);
                snapshot.write(header.constData(), header.size());
                snapshot.write(reinterpret_cast<const char *>(frame->pixels.data()),
                               static_cast<std::streamsize>(frame->pixels.size()));
                snapshot.close();
                if (snapshot && std::rename(temporaryPath.c_str(), outputPath.c_str()) == 0) {
                    QByteArray request(28, '\0');
                    std::memcpy(request.data(), "MVR2", 4);
                    qToLittleEndian<quint64>(frame->id, reinterpret_cast<uchar *>(request.data() + 4));
                    qToLittleEndian<quint64>(static_cast<quint64>(frame->producedNs), reinterpret_cast<uchar *>(request.data() + 12));
                    const float inputRate = static_cast<float>(fps);
                    const float submittedRate = static_cast<float>(consumed_fps_.load());
                    quint32 inputRateBits = 0;
                    quint32 submittedRateBits = 0;
                    std::memcpy(&inputRateBits, &inputRate, sizeof(inputRateBits));
                    std::memcpy(&submittedRateBits, &submittedRate, sizeof(submittedRateBits));
                    qToLittleEndian<quint32>(inputRateBits, reinterpret_cast<uchar *>(request.data() + 20));
                    qToLittleEndian<quint32>(submittedRateBits, reinterpret_cast<uchar *>(request.data() + 24));
                    const unsigned char selection[6] = {
                        'M', 'V', 'S', '1',
                        static_cast<unsigned char>(ai_road_quadrant_.load()),
                        static_cast<unsigned char>(ai_navigation_quadrant_.load()),
                    };
                    (void)::sendto(socketFd, selection, sizeof(selection), 0,
                                   reinterpret_cast<const sockaddr *>(&requestAddress),
                                   sizeof(requestAddress));
                    if (sendto(socketFd, request.constData(), request.size(), 0,
                               reinterpret_cast<const sockaddr *>(&requestAddress), sizeof(requestAddress)) == request.size()) {
                        ai_requests_sent_.fetch_add(1);
                    }
                }
            }
            nextRequest += interval;
            if (nextRequest <= now) nextRequest = now + interval;
        }

        while (true) {
            const ssize_t received = recvfrom(socketFd, packet.data(), packet.size(), 0, nullptr, nullptr);
            if (received < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) break;
                break;
            }
            if (received < 40 || std::memcmp(packet.data(), "MVA2", 4) != 0) continue;
            const uint64_t frameId = qFromLittleEndian<quint64>(packet.data() + 4);
            const uint64_t timestampNs = qFromLittleEndian<quint64>(packet.data() + 12);
            const uint32_t count = qFromLittleEndian<quint32>(packet.data() + 20);
            const uint32_t statusBytes = qFromLittleEndian<quint32>(packet.data() + 36);
            constexpr size_t headerBytes = 40;
            constexpr size_t boxBytes = 36;
            if (count > 32 || statusBytes > 4096 ||
                static_cast<size_t>(received) != headerBytes + count * boxBytes + statusBytes ||
                frameId <= latestResultId || frameId > source_frame_id_.load()) continue;
            const qint64 nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();

            if (nowNs < static_cast<qint64>(timestampNs) || nowNs - static_cast<qint64>(timestampNs) > 250000000) continue;

            QVariantList detectionMaps;
            for (uint32_t index = 0; index < count; ++index) {
                const unsigned char *box = packet.data() + headerBytes + index * boxBytes;
                float values[7]{};
                std::memcpy(values, box, sizeof(values));
                const uint32_t classId = qFromLittleEndian<quint32>(box + 28);
                float confidence = 0.0f;
                uint32_t confidenceBits = qFromLittleEndian<quint32>(box + 32);
                std::memcpy(&confidence, &confidenceBits, sizeof(confidence));
                bool valid = std::isfinite(confidence);
                for (float value : values) valid = valid && std::isfinite(value);
                if (!valid || values[2] <= values[0] || values[3] <= values[1]) continue;
                QVariantMap item;
                item.insert(QStringLiteral("x"), std::clamp(values[0], 0.0f, 1280.0f));
                item.insert(QStringLiteral("y"), std::clamp(values[1], 0.0f, 720.0f));
                item.insert(QStringLiteral("width"), std::clamp(values[2] - values[0], 0.0f, 1280.0f));
                item.insert(QStringLiteral("height"), std::clamp(values[3] - values[1], 0.0f, 720.0f));
                item.insert(QStringLiteral("color"), QColor::fromRgbF(
                    std::clamp(values[4], 0.0f, 1.0f), std::clamp(values[5], 0.0f, 1.0f),
                    std::clamp(values[6], 0.0f, 1.0f)).name());
                const QString className = classNameForId(classId);
                item.insert(QStringLiteral("className"), className);
                item.insert(QStringLiteral("classId"), static_cast<int>(classId));
                item.insert(QStringLiteral("confidence"), static_cast<double>(confidence));
                detectionMaps.append(item);
            }

            QVariantList boxes;
            {
                QMutexLocker locker(&ai_metrics_mutex_);
                const uint32_t resultEpoch = ai_epoch_.load();
                if (resultEpoch != last_ai_epoch_) {
                    tracks_.clear();
                    next_track_id_ = 1;
                    last_ai_epoch_ = resultEpoch;
                }
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
            writeTeslaSceneTracks(boxes, frameId, static_cast<qint64>(timestampNs));
            const QByteArray statusData(reinterpret_cast<const char *>(packet.data() + headerBytes + count * boxBytes),
                                        static_cast<int>(statusBytes));
            const QJsonObject status = QJsonDocument::fromJson(statusData).object();
            const QString traffic = status.value(QStringLiteral("traffic")).toString(QStringLiteral("UNKNOWN"));
            const QJsonArray hazards = status.value(QStringLiteral("hazards")).toArray();
            const QJsonObject navigation = status.value(QStringLiteral("navigation")).toObject();
            const QString direction = navigation.value(QStringLiteral("direction")).toString(QStringLiteral("等待识别"));
            const QString navigationRoad = navigation.value(QStringLiteral("road_name")).toString();
            const QString navigationState = navigation.value(QStringLiteral("state")).toString(QStringLiteral("UNKNOWN"));
            const QString navigationDirection = navigation.value(QStringLiteral("direction")).toString(QStringLiteral("none"));
            const double navigationDistance =
                navigation.value(QStringLiteral("distance_m")).isDouble()
                    ? navigation.value(QStringLiteral("distance_m")).toDouble() : -1.0;
            QString navigationText = direction;
            if (navigation.value(QStringLiteral("distance_m")).isDouble()) {
                navigationText += QStringLiteral(" · %1 m").arg(navigation.value(QStringLiteral("distance_m")).toDouble(), 0, 'f', 0);
            }
            QVariantList hazardItems;
            for (const QJsonValue &hazard : hazards) {
                if (hazard.isString()) hazardItems.append(hazard.toString());
            }
            QVariantMap counts;
            const QJsonObject detections = status.value(QStringLiteral("detections")).toObject();
            for (auto it = detections.begin(); it != detections.end(); ++it) {
                counts.insert(it.key(), it.value().toInt());
            }
            const QString summary = QStringLiteral("AI 帧 %1 · %2 · %3 框 · %4 危险")
                                        .arg(frameId).arg(traffic).arg(boxes.size()).arg(hazards.size());
            {
                QMutexLocker locker(&ai_metrics_mutex_);
                ai_boxes_ = boxes;
                ai_hazards_ = hazardItems;
                ai_status_ = summary;
                ai_traffic_ = traffic;
                ai_navigation_ = navigationText;
                ai_navigation_road_ = navigationRoad;
                ai_navigation_state_ = navigationState;
                ai_navigation_direction_ = navigationDirection;
                ai_navigation_distance_m_ = navigationDistance;
                ai_detection_counts_ = counts;
                ai_fps_.store(status.value(QStringLiteral("ai_fps")).toDouble());
                const QJsonValue hazardAge = status.value(QStringLiteral("hazard_age_ms"));
                if (hazardAge.isDouble()) ai_hazard_age_ms_.store(qRound(hazardAge.toDouble()));
                else ai_hazard_age_ms_.store(0);
                latest_ai_frame_id_.store(frameId);
                const uint64_t displayId = source_frame_id_.load();
                ai_frame_lag_.store(displayId > frameId
                                        ? static_cast<int>(std::min<uint64_t>(displayId - frameId, 99))
                                        : 0);
            }
            last_ai_result_ns_.store(nowNs);
            ai_online_.store(true);
            ai_dirty_.store(true);
            ai_results_received_.fetch_add(1);
            if (ai_timing_stream_.is_open()) {
                const qint64 applyNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                ai_timing_stream_ << "{\"frame_id\":" << frameId
                                  << ",\"qt_receive_ns\":" << nowNs
                                  << ",\"qt_apply_ns\":" << applyNs
                                  << ",\"display_frame_id\":" << source_frame_id_.load()
                                  << "}" << std::endl;
            }
            latestResultId = frameId;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    close(socketFd);
}

void VideoItem::requestFrameUpdate() {
    update_queued_.store(false);
    emit frameAdvanced();
    update();
}



QString VideoItem::aiStatus() const {
    QMutexLocker locker(&ai_metrics_mutex_);
    return ai_status_;
}

QString VideoItem::aiTraffic() const {
    QMutexLocker locker(&ai_metrics_mutex_);
    return ai_traffic_;
}

QString VideoItem::aiNavigation() const {
    QMutexLocker locker(&ai_metrics_mutex_);
    return ai_navigation_;
}

QString VideoItem::aiNavigationRoad() const {
    QMutexLocker locker(&ai_metrics_mutex_);
    return ai_navigation_road_;
}

QString VideoItem::aiNavigationState() const {
    QMutexLocker locker(&ai_metrics_mutex_);
    return ai_navigation_state_;
}

QString VideoItem::aiNavigationDirection() const {
    QMutexLocker locker(&ai_metrics_mutex_);
    return ai_navigation_direction_;
}

double VideoItem::aiNavigationDistanceM() const {
    QMutexLocker locker(&ai_metrics_mutex_);
    return ai_navigation_distance_m_;
}

int VideoItem::aiDetectionCount() const {
    QMutexLocker locker(&ai_metrics_mutex_);
    return ai_boxes_.size();
}

QVariantMap VideoItem::aiDetectionCounts() const {
    QMutexLocker locker(&ai_metrics_mutex_);
    return ai_detection_counts_;
}

double VideoItem::sourceAgeMs() const {
    const qint64 last = last_source_ns_.load();
    if (last <= 0) return 0.0;
    return std::max<qint64>(0, steadyNs() - last) / 1.0e6;
}

QRectF VideoItem::sourceViewport() const {
    QMutexLocker locker(&viewport_mutex_);
    return source_viewport_;
}

void VideoItem::setSourceViewport(const QRectF &viewport) {
    QRectF normalized = viewport.normalized();
    normalized.setLeft(std::clamp(normalized.left(), 0.0, 1.0));
    normalized.setTop(std::clamp(normalized.top(), 0.0, 1.0));
    normalized.setRight(std::clamp(normalized.right(), 0.0, 1.0));
    normalized.setBottom(std::clamp(normalized.bottom(), 0.0, 1.0));
    if (normalized.width() <= 0.001 || normalized.height() <= 0.001) {
        normalized = QRectF(0.0, 0.0, 1.0, 1.0);
    }
    {
        QMutexLocker locker(&viewport_mutex_);
        if (source_viewport_ == normalized) return;
        source_viewport_ = normalized;
    }
    // The FBO renderer only copies the viewport during synchronize(); mark the
    // item dirty so a mode change is visible even when the video texture is
    // otherwise static.
    update();
    emit sourceViewportChanged();
}

void VideoItem::sendAiSelection(int road, int navigation) {
    const int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(38260);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const unsigned char packet[6] = {
        'M', 'V', 'S', '1',
        static_cast<unsigned char>(std::clamp(road, 0, 3)),
        static_cast<unsigned char>(std::clamp(navigation, 0, 3)),
    };
    (void)::sendto(fd, packet, sizeof(packet), 0,
                   reinterpret_cast<const sockaddr *>(&address), sizeof(address));
    close(fd);
}

void VideoItem::setSpatialViewActive(bool value) {
    if (spatial_view_active_.exchange(value) == value) return;
    QFile flag(QStringLiteral("/dev/shm/tesla_scene.active"));
    if (flag.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        flag.write(value ? "1\n" : "0\n");
    }
    emit spatialViewActiveChanged();
}

void VideoItem::writeTeslaSceneTracks(const QVariantList &boxes, uint64_t frameId,
                                      qint64 timestampNs) {
    if (!spatial_view_active_.load()) return;
    QJsonObject root;
    root.insert(QStringLiteral("source"), source_mode_);
    root.insert(QStringLiteral("source_frame_id"), static_cast<qint64>(frameId));
    root.insert(QStringLiteral("timestamp_ns"), static_cast<qint64>(timestampNs));
    root.insert(QStringLiteral("display_width"), kSourceWidth);
    root.insert(QStringLiteral("display_height"), kSourceHeight);
    root.insert(QStringLiteral("simulated"), false);
    QJsonArray items;
    for (const QVariant &entry : boxes) {
        const QVariantMap box = entry.toMap();
        if (box.isEmpty()) continue;
        QJsonObject item;
        item.insert(QStringLiteral("id"), static_cast<qint64>(box.value(QStringLiteral("trackId")).toULongLong()));
        item.insert(QStringLiteral("class"), box.value(QStringLiteral("className")).toString());
        item.insert(QStringLiteral("x"), box.value(QStringLiteral("x")).toDouble());
        item.insert(QStringLiteral("y"), box.value(QStringLiteral("y")).toDouble());
        item.insert(QStringLiteral("width"), box.value(QStringLiteral("width")).toDouble());
        item.insert(QStringLiteral("height"), box.value(QStringLiteral("height")).toDouble());
        item.insert(QStringLiteral("vx"), box.value(QStringLiteral("vx")).toDouble());
        item.insert(QStringLiteral("vy"), box.value(QStringLiteral("vy")).toDouble());
        item.insert(QStringLiteral("confidence"), box.value(QStringLiteral("confidence")).toDouble());
        item.insert(QStringLiteral("state"), box.value(QStringLiteral("state")).toString());
        item.insert(QStringLiteral("detect_timestamp_ns"),
                    static_cast<qint64>(box.value(QStringLiteral("detectTimestampNs")).toULongLong()));
        items.append(item);
    }
    root.insert(QStringLiteral("tracks"), items);
    const QByteArray payload = QJsonDocument(root).toJson(QJsonDocument::Compact);
    const QString target = QStringLiteral("/dev/shm/tesla_scene_tracks.json");
    const QString temporary = target + QStringLiteral(".tmp");
    QFile file(temporary);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) return;
    file.write(payload);
    file.close();
    QFile::remove(target);
    QFile::rename(temporary, target);
}

void VideoItem::setAiRoadQuadrant(int quadrant) {
    quadrant = std::clamp(quadrant, 0, 3);
    if (ai_road_quadrant_.exchange(quadrant) == quadrant) return;
    {
        QMutexLocker locker(&ai_metrics_mutex_);
        ai_boxes_.clear();
        tracks_.clear();
        next_track_id_ = 1;
        ai_hazards_.clear();
        ai_detection_counts_.clear();
        ai_traffic_ = QStringLiteral("UNKNOWN");
        ai_navigation_ = QStringLiteral("等待导航识别");
        ai_navigation_road_.clear();
        ai_navigation_state_ = QStringLiteral("UNKNOWN");
        ai_status_ = QStringLiteral("切换主识别源 Q%1").arg(quadrant + 1);
    }
    last_ai_result_ns_.store(0);
    ai_online_.store(false);
    ai_dirty_.store(true);
    sendAiSelection(quadrant, ai_navigation_quadrant_.load());
    emit aiSelectionChanged();
}

void VideoItem::setAiNavigationQuadrant(int quadrant) {
    quadrant = std::clamp(quadrant, 0, 3);
    if (ai_navigation_quadrant_.exchange(quadrant) == quadrant) return;
    {
        QMutexLocker locker(&ai_metrics_mutex_);
        ai_navigation_ = QStringLiteral("等待导航识别");
        ai_navigation_road_.clear();
        ai_navigation_state_ = QStringLiteral("UNKNOWN");
        ai_navigation_direction_ = QStringLiteral("none");
        ai_navigation_distance_m_ = -1.0;
    }
    ai_dirty_.store(true);
    sendAiSelection(ai_road_quadrant_.load(), quadrant);
    emit aiSelectionChanged();
}

QVariantList VideoItem::aiBoxes() const {
    QMutexLocker locker(&ai_metrics_mutex_);
    return ai_boxes_;
}

QVariantList VideoItem::aiHazards() const {
    QMutexLocker locker(&ai_metrics_mutex_);
    return ai_hazards_;
}

void VideoItem::updateMetrics() {
    // Windowed counters: restart() gives the actual interval since the last
    // sample. Do not divide cumulative counts by total process uptime.
    const double seconds = std::max(0.001, stats_clock_.restart() / 1000.0);
    const uint64_t produced = produced_frames_.load();
    const uint64_t consumed = render_metrics_->uploaded_frames.load();
    const uint64_t rendered = render_metrics_->rendered_frames.load();
    input_fps_.store((produced - last_produced_for_stats_) / seconds);
    consumed_fps_.store((consumed - last_consumed_for_stats_) / seconds);
    render_metrics_->render_fps.store((rendered - last_rendered_for_stats_) / seconds);
    render_metrics_->synchronize_fps.store((render_metrics_->synchronize_calls.load() - last_sync_for_stats_) / seconds);
    last_sync_for_stats_ = render_metrics_->synchronize_calls.load();
    last_produced_for_stats_ = produced;
    last_consumed_for_stats_ = consumed;
    last_rendered_for_stats_ = rendered;
    const uint64_t cpuTicks = processCpuTicks();
    const long ticksPerSecond = sysconf(_SC_CLK_TCK);
    const uint64_t cpuDelta = cpuTicks >= last_cpu_ticks_ ? cpuTicks - last_cpu_ticks_ : 0;
    cpu_percent_.store(last_cpu_ticks_ && ticksPerSecond > 0
                           ? 100.0 * cpuDelta / (ticksPerSecond * seconds * std::max(1, QThread::idealThreadCount()))
                           : 0.0);
    last_cpu_ticks_ = cpuTicks;
    rss_mb_.store(processRssMb());
    double gpuLoad = gpu_load_percent_.load();
    uint64_t gpuFrequency = gpu_frequency_hz_.load();
    readGpuCounters(&gpuLoad, &gpuFrequency);
    gpu_load_percent_.store(gpuLoad);
    gpu_frequency_hz_.store(gpuFrequency);
    double npuLoad = npu_load_percent_.load();
    readNpuCounter(&npuLoad);
    npu_load_percent_.store(npuLoad);
    source_fps_.store(input_fps_.load());

    const qint64 nowNs = steadyNs();
    if (source_mode_.startsWith(QStringLiteral("pgv"))) {
        const qint64 lastSource = last_source_ns_.load();
        const bool online = lastSource > 0 && nowNs - lastSource < 500000000LL;
        source_online_.store(online);
    }

    const qint64 lastAi = last_ai_result_ns_.load();
    const qint64 aiAgeNs = lastAi > 0 ? std::max<qint64>(0, nowNs - lastAi) : 0;
    const bool aiOnlineNow = ai_running_.load() && lastAi > 0 && aiAgeNs < 1500000000LL;
    ai_result_age_ms_.store(static_cast<int>(aiAgeNs / 1000000));
    if (ai_online_.load() && !aiOnlineNow) {
        QMutexLocker locker(&ai_metrics_mutex_);
        ai_boxes_.clear();
        tracks_.clear();
        next_track_id_ = 1;
        ai_hazards_.clear();
        ai_detection_counts_.clear();
        ai_traffic_ = QStringLiteral("UNKNOWN");
        ai_navigation_ = QStringLiteral("等待导航识别");
        ai_navigation_road_.clear();
        ai_navigation_state_ = QStringLiteral("UNKNOWN");
        ai_navigation_direction_ = QStringLiteral("none");
        ai_navigation_distance_m_ = -1.0;
        ai_status_ = QStringLiteral("AI sidecar 未连接");
        ai_fps_.store(0.0);
        ai_dirty_.store(true);
    }
    ai_online_.store(aiOnlineNow);
    if (ai_dirty_.exchange(false)) {
        emit aiChanged();
    }

    last_produced_for_stats_ = produced;
    last_consumed_for_stats_ = consumed;
    last_rendered_for_stats_ = rendered;
    last_cpu_ticks_ = cpuTicks;
    stats_clock_.restart();
    emit metricsChanged();
}

void VideoItem::recordExternalUpload(const Frame &frame, double uploadMs, double sourceAgeMs) {
    render_metrics_->upload_ms.store(uploadMs);
    render_metrics_->source_to_upload_ms.store(sourceAgeMs);
    render_metrics_->frame_id.store(frame.id);
    render_metrics_->uploaded_frames.fetch_add(1);
    {
        QMutexLocker contentLocker(&render_metrics_->replay_content_mutex);
        const quint64 sourceIndex = static_cast<quint64>(frame.sourceIndex);
        if (!render_metrics_->replay_content_indices.contains(sourceIndex)) {
            render_metrics_->replay_content_indices.insert(sourceIndex);
            render_metrics_->unique_replay_content_frames.fetch_add(1);
        }
    }
    {
        QMutexLocker locker(&render_metrics_->upload_mutex);
        render_metrics_->upload_samples.push_back(uploadMs);
        if (render_metrics_->upload_samples.size() > 3600)
            render_metrics_->upload_samples.erase(render_metrics_->upload_samples.begin(), render_metrics_->upload_samples.begin() + 600);
        render_metrics_->source_age_samples.push_back(sourceAgeMs);
        if (render_metrics_->source_age_samples.size() > 3600)
            render_metrics_->source_age_samples.erase(render_metrics_->source_age_samples.begin(), render_metrics_->source_age_samples.begin() + 600);
    }
    noteRenderedFrame(frame.id, frame.sourceTimestampNs > 0 ? frame.sourceTimestampNs : frame.producedNs);
}

double VideoItem::uploadP95Ms() const {
    QMutexLocker locker(&render_metrics_->upload_mutex);
    if (render_metrics_->upload_samples.empty()) return 0.0;
    auto samples = render_metrics_->upload_samples;
    std::sort(samples.begin(), samples.end());
    return samples[static_cast<size_t>(0.95 * (samples.size() - 1))];
}

double VideoItem::sourceToUploadP50Ms() const {
    QMutexLocker locker(&render_metrics_->age_mutex);
    if (render_metrics_->source_age_samples.empty()) return 0.0;
    auto samples = render_metrics_->source_age_samples;
    std::sort(samples.begin(), samples.end());
    return samples[static_cast<size_t>(0.50 * (samples.size() - 1))];
}

double VideoItem::sourceToUploadP95Ms() const {
    QMutexLocker locker(&render_metrics_->age_mutex);
    if (render_metrics_->source_age_samples.empty()) return 0.0;
    auto samples = render_metrics_->source_age_samples;
    std::sort(samples.begin(), samples.end());
    return samples[static_cast<size_t>(0.95 * (samples.size() - 1))];
}

QString VideoItem::renderer() const {
    QMutexLocker locker(&render_metrics_->strings_mutex);
    return render_metrics_->renderer;
}
QString VideoItem::vendor() const {
    QMutexLocker locker(&render_metrics_->strings_mutex);
    return render_metrics_->vendor;
}
QString VideoItem::glVersion() const {
    QMutexLocker locker(&render_metrics_->strings_mutex);
    return render_metrics_->gl_version;
}
QString VideoItem::eglVendor() const {
    QMutexLocker locker(&render_metrics_->strings_mutex);
    return render_metrics_->egl_vendor;
}
