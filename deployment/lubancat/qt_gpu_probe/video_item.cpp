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
#include <QFile>
#include <QMetaObject>
#include <QThread>
#include <thread>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <cstdio>
#include <unistd.h>

namespace {
constexpr int kSourceWidth = 1280;
constexpr int kSourceHeight = 720;
constexpr size_t kFrameBytes = static_cast<size_t>(kSourceWidth) * kSourceHeight * 2;

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
    explicit Renderer(std::shared_ptr<VideoItem::RenderMetrics> metrics)
        : metrics_(std::move(metrics)) {}

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
        if (!pending_) pending_ = video->takeLatest();
    }

    void render() override {
        metrics_->rendered_frames.fetch_add(1);
        if (!program_) initGL();
        if (!program_) return;
        auto *context = QOpenGLContext::currentContext();
        if (!context) return;
        auto *f = context->functions();
        const auto frame = pending_;
        if (frame && frame->id != uploaded_id_) {
            QElapsedTimer timer;
            timer.start();
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
                f->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, kSourceWidth, kSourceHeight, 0,
                                GL_RGB, GL_UNSIGNED_SHORT_5_6_5, frame->pixels.data());
            } else {
                f->glBindTexture(GL_TEXTURE_2D, texture_);
                f->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kSourceWidth, kSourceHeight,
                                   GL_RGB, GL_UNSIGNED_SHORT_5_6_5, frame->pixels.data());
            }
            const GLenum uploadError = f->glGetError();
            if (uploadError == GL_NO_ERROR) {
                uploaded_id_ = frame->id;
                const double uploadMs = static_cast<double>(timer.nsecsElapsed()) / 1.0e6;
                const qint64 nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                const double sourceAgeMs = std::max<qint64>(0, nowNs - frame->producedNs) / 1.0e6;
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
        const GLfloat vertices[] = {
            -1.f, -1.f, 0.f, 1.f,  1.f, -1.f, 1.f, 1.f,
            -1.f,  1.f, 0.f, 0.f,  1.f,  1.f, 1.f, 0.f,
        };
        program_->enableAttributeArray(0);
        program_->setAttributeArray(0, GL_FLOAT, vertices, 2, 4 * sizeof(GLfloat));
        program_->enableAttributeArray(1);
        program_->setAttributeArray(1, GL_FLOAT, vertices + 2, 2, 4 * sizeof(GLfloat));
        f->glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        program_->disableAttributeArray(0);
        program_->disableAttributeArray(1);
        program_->release();
        update();
    }

private:
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
    uint64_t uploaded_id_ = UINT64_MAX;
    GLuint texture_ = 0;
    std::unique_ptr<QOpenGLShaderProgram> program_;
};

VideoItem::VideoItem(QQuickItem *parent) : QQuickFramebufferObject(parent) {
    setMirrorVertically(false);
    stats_clock_.start();
    connect(&metrics_timer_, &QTimer::timeout, this, &VideoItem::updateMetrics);
    metrics_timer_.start(500);
}

VideoItem::~VideoItem() { stopProducer(); }

QQuickFramebufferObject::Renderer *VideoItem::createRenderer() const {
    return new Renderer(render_metrics_);
}

void VideoItem::startProducer(double fps, const QStringList &replayFrames,
                              const QString &aiFrameFile, double aiHz) {
    stopProducer();
    replay_frames_ = replayFrames;
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
    {
        QMutexLocker locker(&frame_mutex_);
        latest_frame_ = frame;
    }
    if (ai_running_.load()) {
        std::lock_guard<std::mutex> locker(ai_frame_mutex_);
        latest_ai_frame_ = std::move(frame);
    }
    if (!update_queued_.exchange(true) &&
        !QMetaObject::invokeMethod(this, "requestFrameUpdate", Qt::QueuedConnection)) {
        update_queued_.store(false);
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
                frameId <= latestResultId || frameId > produced_frames_.load()) continue;
            const qint64 nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (nowNs < static_cast<qint64>(timestampNs) || nowNs - static_cast<qint64>(timestampNs) > 250000000) continue;

            QVariantList boxes;
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
                item.insert(QStringLiteral("label"), QStringLiteral("ID %1  %2%").arg(classId).arg(qRound(confidence * 100.0f)));
                boxes.append(item);
            }
            const QByteArray statusData(reinterpret_cast<const char *>(packet.data() + headerBytes + count * boxBytes),
                                        static_cast<int>(statusBytes));
            const QJsonObject status = QJsonDocument::fromJson(statusData).object();
            const QString traffic = status.value(QStringLiteral("traffic")).toString(QStringLiteral("UNKNOWN"));
            const QJsonArray hazards = status.value(QStringLiteral("hazards")).toArray();
            const QJsonObject navigation = status.value(QStringLiteral("navigation")).toObject();
            const QString direction = navigation.value(QStringLiteral("direction")).toString(QStringLiteral("等待识别"));
            QString navigationText = direction;
            if (navigation.value(QStringLiteral("distance_m")).isDouble()) {
                navigationText += QStringLiteral(" · %1 m").arg(navigation.value(QStringLiteral("distance_m")).toDouble(), 0, 'f', 0);
            }
            QVariantList hazardItems;
            for (const QJsonValue &hazard : hazards) {
                if (hazard.isString()) hazardItems.append(hazard.toString());
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
                latest_ai_frame_id_.store(frameId);
            }
            ai_results_received_.fetch_add(1);
            latestResultId = frameId;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    close(socketFd);
}

void VideoItem::requestFrameUpdate() {
    update_queued_.store(false);
    update();
}

std::shared_ptr<VideoItem::Frame> VideoItem::takeLatest() {
    QMutexLocker locker(&frame_mutex_);
    if (!latest_frame_) return {};
    auto frame = std::move(latest_frame_);
    return frame;
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

int VideoItem::aiDetectionCount() const {
    QMutexLocker locker(&ai_metrics_mutex_);
    return ai_boxes_.size();
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
    const double seconds = std::max(0.001, stats_clock_.elapsed() / 1000.0);
    const uint64_t produced = produced_frames_.load();
    const uint64_t consumed = render_metrics_->uploaded_frames.load();
    const uint64_t rendered = render_metrics_->rendered_frames.load();
    input_fps_.store((produced - last_produced_for_stats_) / seconds);
    consumed_fps_.store((consumed - last_consumed_for_stats_) / seconds);
    render_metrics_->render_fps.store((rendered - last_rendered_for_stats_) / seconds);
    const uint64_t cpuTicks = processCpuTicks();
    const long ticksPerSecond = sysconf(_SC_CLK_TCK);
    const uint64_t cpuDelta = cpuTicks >= last_cpu_ticks_ ? cpuTicks - last_cpu_ticks_ : 0;
    cpu_percent_.store(last_cpu_ticks_ && ticksPerSecond > 0
                           ? 100.0 * cpuDelta / (ticksPerSecond * seconds * std::max(1, QThread::idealThreadCount()))
                           : 0.0);
    rss_mb_.store(processRssMb());
    double gpuLoad = gpu_load_percent_.load();
    uint64_t gpuFrequency = gpu_frequency_hz_.load();
    readGpuCounters(&gpuLoad, &gpuFrequency);
    gpu_load_percent_.store(gpuLoad);
    gpu_frequency_hz_.store(gpuFrequency);
    last_produced_for_stats_ = produced;
    last_consumed_for_stats_ = consumed;
    last_rendered_for_stats_ = rendered;
    last_cpu_ticks_ = cpuTicks;
    stats_clock_.restart();
    emit metricsChanged();
}

double VideoItem::uploadP95Ms() const {
    QMutexLocker locker(&render_metrics_->upload_mutex);
    if (render_metrics_->upload_samples.empty()) return 0.0;
    auto samples = render_metrics_->upload_samples;
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
