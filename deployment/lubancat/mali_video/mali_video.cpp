#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#include <arpa/inet.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "The PCIe RGB565 source contract is little-endian."
#endif

using EGLint = int;
using EGLBoolean = unsigned int;
using EGLenum = unsigned int;
using EGLDisplay = void *;
using EGLConfig = void *;
using EGLContext = void *;
using EGLSurface = void *;
using EGLNativeDisplayType = Display *;
using EGLNativeWindowType = unsigned long;
using GLenum = unsigned int;
using GLuint = unsigned int;
using GLint = int;
using GLsizei = int;
using GLboolean = unsigned char;
using GLfloat = float;
using GLchar = char;

extern "C" {
EGLDisplay eglGetDisplay(EGLNativeDisplayType);
EGLBoolean eglInitialize(EGLDisplay, EGLint *, EGLint *);
EGLBoolean eglChooseConfig(EGLDisplay, const EGLint *, EGLConfig *, EGLint, EGLint *);
EGLBoolean eglGetConfigAttrib(EGLDisplay, EGLConfig, EGLint, EGLint *);
EGLBoolean eglBindAPI(EGLenum);
EGLContext eglCreateContext(EGLDisplay, EGLConfig, EGLContext, const EGLint *);
EGLSurface eglCreateWindowSurface(EGLDisplay, EGLConfig, EGLNativeWindowType, const EGLint *);
EGLBoolean eglMakeCurrent(EGLDisplay, EGLSurface, EGLSurface, EGLContext);
EGLBoolean eglSwapInterval(EGLDisplay, EGLint);
EGLBoolean eglSwapBuffers(EGLDisplay, EGLSurface);
EGLBoolean eglDestroySurface(EGLDisplay, EGLSurface);
EGLBoolean eglDestroyContext(EGLDisplay, EGLContext);
EGLBoolean eglTerminate(EGLDisplay);
EGLint eglGetError(void);
const char *eglQueryString(EGLDisplay, EGLint);

const unsigned char *glGetString(GLenum);
GLuint glCreateShader(GLenum);
void glShaderSource(GLuint, GLsizei, const GLchar *const *, const GLint *);
void glCompileShader(GLuint);
void glGetShaderiv(GLuint, GLenum, GLint *);
void glGetShaderInfoLog(GLuint, GLsizei, GLsizei *, GLchar *);
void glDeleteShader(GLuint);
GLuint glCreateProgram(void);
void glAttachShader(GLuint, GLuint);
void glBindAttribLocation(GLuint, GLuint, const GLchar *);
void glLinkProgram(GLuint);
void glGetProgramiv(GLuint, GLenum, GLint *);
void glGetProgramInfoLog(GLuint, GLsizei, GLsizei *, GLchar *);
void glDeleteProgram(GLuint);
void glUseProgram(GLuint);
GLint glGetAttribLocation(GLuint, const GLchar *);
GLint glGetUniformLocation(GLuint, const GLchar *);
void glGenTextures(GLsizei, GLuint *);
void glDeleteTextures(GLsizei, const GLuint *);
void glBindTexture(GLenum, GLuint);
void glTexParameteri(GLenum, GLenum, GLint);
void glPixelStorei(GLenum, GLint);
void glTexImage2D(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void *);
void glTexSubImage2D(GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void *);
void glViewport(GLint, GLint, GLsizei, GLsizei);
void glScissor(GLint, GLint, GLsizei, GLsizei);
void glClearColor(GLfloat, GLfloat, GLfloat, GLfloat);
void glClear(GLenum);
void glVertexAttribPointer(GLuint, GLint, GLenum, GLboolean, GLsizei, const void *);
void glEnableVertexAttribArray(GLuint);
void glEnable(GLenum);
void glDisable(GLenum);
void glBlendFunc(GLenum, GLenum);
void glLineWidth(GLfloat);
void glActiveTexture(GLenum);
void glUniform1i(GLint, GLint);
void glUniform4f(GLint, GLfloat, GLfloat, GLfloat, GLfloat);
void glDrawArrays(GLenum, GLint, GLsizei);
GLenum glGetError(void);
void glFinish(void);
void glReadPixels(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void *);
}

namespace {
using Clock = std::chrono::steady_clock;
constexpr uint32_t kWidth = 1280;
constexpr uint32_t kHeight = 720;
constexpr uint32_t kBytesPerPixel = 2;
constexpr uint32_t kStrideBytes = kWidth * kBytesPerPixel;
constexpr size_t kFrameBytes = static_cast<size_t>(kStrideBytes) * kHeight;
constexpr uint32_t kPgvDmaAlignment = 4096;
constexpr uint32_t kPgvBufferBytes = 1847296;
constexpr uint32_t kSyntheticPoolFrames = 60;
constexpr size_t kMaxDetectionBoxes = 32;

struct UiRect {
  GLfloat x;
  GLfloat y;
  GLfloat width;
  GLfloat height;
};

// Frozen 1280x720 composition shared by the design preview and GPU clip regions.
struct LayoutSpec {
  UiRect main_video{14.f, 62.f, 986.f, 468.f};
  UiRect signal_card{24.f, 78.f, 300.f, 182.f};
  UiRect navigation_card{1004.f, 78.f, 252.f, 136.f};
  UiRect hazard_card{1004.f, 224.f, 252.f, 148.f};
  UiRect lane_card{1004.f, 382.f, 252.f, 148.f};
  std::array<UiRect, 4> thumbnails{{
      {16.f, 548.f, 204.f, 150.f},
      {230.f, 548.f, 204.f, 150.f},
      {444.f, 548.f, 204.f, 150.f},
      {658.f, 548.f, 204.f, 150.f},
  }};
  UiRect spatial_card{872.f, 548.f, 392.f, 150.f};
};

constexpr LayoutSpec kLayout{};

static_assert(kFrameBytes == 1843200, "PGV1 frame byte count changed.");
static_assert((kFrameBytes + 32 + kPgvDmaAlignment - 1) / kPgvDmaAlignment *
                  kPgvDmaAlignment ==
              kPgvBufferBytes,
              "PGV1 coherent DMA slot layout changed.");

constexpr EGLint EGL_NONE = 0x3038;
constexpr EGLint EGL_SURFACE_TYPE = 0x3033;
constexpr EGLint EGL_WINDOW_BIT = 0x0004;
constexpr EGLint EGL_RENDERABLE_TYPE = 0x3040;
constexpr EGLint EGL_OPENGL_ES2_BIT = 0x0004;
constexpr EGLint EGL_RED_SIZE = 0x3024;
constexpr EGLint EGL_GREEN_SIZE = 0x3023;
constexpr EGLint EGL_BLUE_SIZE = 0x3022;
constexpr EGLint EGL_ALPHA_SIZE = 0x3021;
constexpr EGLint EGL_NATIVE_VISUAL_ID = 0x302e;
constexpr EGLint EGL_CONTEXT_CLIENT_VERSION = 0x3098;
constexpr EGLenum EGL_OPENGL_ES_API = 0x30a0;
constexpr EGLint EGL_VENDOR = 0x3053;
constexpr EGLint EGL_VERSION = 0x3054;

constexpr GLenum GL_VERTEX_SHADER = 0x8b31;
constexpr GLenum GL_FRAGMENT_SHADER = 0x8b30;
constexpr GLenum GL_COMPILE_STATUS = 0x8b81;
constexpr GLenum GL_LINK_STATUS = 0x8b82;
constexpr GLenum GL_INFO_LOG_LENGTH = 0x8b84;
constexpr GLenum GL_TEXTURE_2D = 0x0de1;
constexpr GLenum GL_TEXTURE_MIN_FILTER = 0x2801;
constexpr GLenum GL_TEXTURE_MAG_FILTER = 0x2800;
constexpr GLenum GL_TEXTURE_WRAP_S = 0x2802;
constexpr GLenum GL_TEXTURE_WRAP_T = 0x2803;
constexpr GLint GL_LINEAR = 0x2601;
constexpr GLint GL_CLAMP_TO_EDGE = 0x812f;
constexpr GLenum GL_RGB = 0x1907;
constexpr GLenum GL_RGBA = 0x1908;
constexpr GLenum GL_UNSIGNED_SHORT_5_6_5 = 0x8363;
constexpr GLenum GL_UNSIGNED_BYTE = 0x1401;
constexpr GLenum GL_UNPACK_ALIGNMENT = 0x0cf5;
constexpr GLenum GL_COLOR_BUFFER_BIT = 0x00004000;
constexpr GLenum GL_FLOAT = 0x1406;
constexpr GLenum GL_TRIANGLE_STRIP = 0x0005;
constexpr GLenum GL_TRIANGLES = 0x0004;
constexpr GLenum GL_LINES = 0x0001;
constexpr GLenum GL_BLEND = 0x0be2;
constexpr GLenum GL_SCISSOR_TEST = 0x0c11;
constexpr GLenum GL_SRC_ALPHA = 0x0302;
constexpr GLenum GL_ONE_MINUS_SRC_ALPHA = 0x0303;
constexpr GLenum GL_TEXTURE0 = 0x84c0;
constexpr GLenum GL_NO_ERROR = 0;
constexpr GLenum GL_RENDERER = 0x1f01;
constexpr GLenum GL_VERSION = 0x1f02;

uint64_t monotonicNs() {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC) failed");
  }
  return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL +
         static_cast<uint64_t>(value.tv_nsec);
}

double elapsedMs(uint64_t start, uint64_t end) {
  return static_cast<double>(end - start) / 1000000.0;
}

uint64_t processCpuNs() {
  timespec value{};
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0) return 0;
  return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL +
         static_cast<uint64_t>(value.tv_nsec);
}

struct FrameView {
  const uint16_t *pixels = nullptr;
  uint64_t frame_id = 0;
  uint64_t timestamp_ns = 0;
  uint32_t stride_bytes = kStrideBytes;
  std::shared_ptr<const std::vector<uint16_t>> owner;
};

class FrameSource {
 public:
  virtual ~FrameSource() { stop(); }
  virtual const std::string &name() const = 0;
  virtual const uint16_t *pixelsFor(uint64_t frame_id) const = 0;
  virtual bool acquire(uint64_t source_index, uint64_t scheduled_ns, FrameView &view) {
    const uint16_t *pixels = pixelsFor(source_index);
    if (!pixels) return false;
    view = FrameView{pixels, source_index, scheduled_ns, kStrideBytes, {}};
    return true;
  }
  virtual double prepareMs() const { return 0.0; }

  void start(uint32_t fps, uint32_t duration_seconds) {
    if (fps == 0 || duration_seconds == 0) {
      throw std::invalid_argument("FPS and duration must be positive");
    }
    if (worker_.joinable()) throw std::logic_error("Frame source already started");
    period_ns_ = 1000000000ULL / fps;
    const uint64_t frame_count =
        (static_cast<uint64_t>(duration_seconds) * 1000000000ULL + period_ns_ - 1) /
        period_ns_;
    publish_times_ns_.clear();
    publish_times_ns_.reserve(static_cast<size_t>(frame_count));
    published_.store(0, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(latest_mutex_);
      latest_ = {};
    }
    running_.store(true);
    worker_ = std::thread([this, frame_count]() {
      const auto anchor = Clock::now();
      start_ns_.store(monotonicNs(), std::memory_order_release);
      for (uint64_t id = 0; id < frame_count && running_.load(); ++id) {
        const auto deadline = anchor + std::chrono::nanoseconds(period_ns_ * id);
        std::this_thread::sleep_until(deadline);
        if (!running_.load()) break;
        const uint64_t scheduled_ns =
            start_ns_.load(std::memory_order_acquire) + period_ns_ * id;
        FrameView frame;
        const uint64_t acquire_start_ns = monotonicNs();
        if (!acquire(id, scheduled_ns, frame) || !frame.pixels) continue;
        const uint64_t published_ns = monotonicNs();
        acquire_times_ms_.push_back(elapsedMs(acquire_start_ns, published_ns));
        {
          std::lock_guard<std::mutex> lock(latest_mutex_);
          latest_ = std::move(frame);
        }
        latest_changed_.notify_one();
        publish_times_ns_.push_back(published_ns);
        published_.fetch_add(1, std::memory_order_relaxed);
      }
      running_.store(false);
      latest_changed_.notify_all();
    });
    while (start_ns_.load(std::memory_order_acquire) == 0) std::this_thread::yield();
  }

  bool waitLatest(uint64_t after_frame_id, uint64_t deadline_ns, FrameView &result) const {
    std::unique_lock<std::mutex> lock(latest_mutex_);
    const auto deadline = Clock::time_point(std::chrono::nanoseconds(deadline_ns));
    latest_changed_.wait_until(lock, deadline, [this, after_frame_id]() {
      return (latest_.pixels && latest_.frame_id != after_frame_id) || !running_.load();
    });
    if (!latest_.pixels || latest_.frame_id == after_frame_id) return false;
    result = latest_;
    return true;
  }

  void stop() {
    running_.store(false);
    latest_changed_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

  uint64_t publishedCount() const { return published_.load(std::memory_order_relaxed); }
  uint64_t startNs() const { return start_ns_.load(std::memory_order_acquire); }
  const std::vector<uint64_t> &publishTimesNs() const { return publish_times_ns_; }
  const std::vector<double> &acquireTimesMs() const { return acquire_times_ms_; }

 protected:
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> published_{0};
  std::atomic<uint64_t> start_ns_{0};
  uint64_t period_ns_ = 0;

 private:
  mutable std::mutex latest_mutex_;
  mutable std::condition_variable latest_changed_;
  FrameView latest_{};
  std::thread worker_;
  std::vector<uint64_t> publish_times_ns_;
  std::vector<double> acquire_times_ms_;
};

class ReplayFrameSource final : public FrameSource {
 public:
  explicit ReplayFrameSource(const std::vector<std::string> &paths) {
    if (paths.empty()) throw std::invalid_argument("Replay source needs at least one .rgb565 file");
    for (size_t index = 0; index < paths.size(); ++index) {
      entries_.push_back(ReplayEntry{paths[index], static_cast<uint64_t>(index), 0});
    }
    loadEntries();
  }

  explicit ReplayFrameSource(const std::string &manifest_path) {
    std::ifstream manifest(manifest_path);
    if (!manifest) throw std::runtime_error("Cannot open replay manifest: " + manifest_path);
    const std::string base_dir = [&]() {
      const size_t slash = manifest_path.find_last_of("/");
      return slash == std::string::npos ? std::string(".") : manifest_path.substr(0, slash);
    }();
    std::string line;
    size_t line_number = 0;
    while (std::getline(manifest, line)) {
      ++line_number;
      if (line.empty() || line[0] == '#') continue;
      std::istringstream row(line);
      ReplayEntry entry;
      row >> entry.frame_id >> entry.timestamp_ns >> entry.path;
      if (!row || entry.path.empty()) {
        throw std::runtime_error("Invalid replay manifest row " + std::to_string(line_number));
      }
      if (entry.path.front() != '/') entry.path = base_dir + "/" + entry.path;
      entries_.push_back(std::move(entry));
    }
    if (entries_.empty()) throw std::invalid_argument("Replay manifest has no frames");
    for (size_t i = 1; i < entries_.size(); ++i) {
      if (entries_[i].frame_id <= entries_[i - 1].frame_id ||
          entries_[i].timestamp_ns < entries_[i - 1].timestamp_ns) {
        throw std::invalid_argument("Replay manifest frame IDs and timestamps must be monotonic");
      }
    }
    manifest_path_ = manifest_path;
    loadEntries();
  }

  bool acquire(uint64_t source_index, uint64_t scheduled_ns, FrameView &view) override {
    const ReplayEntry &entry = entries_[source_index % entries_.size()];
    const uint64_t timestamp_ns = manifest_path_.empty()
                                      ? scheduled_ns
                                      : startNs() + entry.timestamp_ns;
    view = FrameView{frames_[source_index % frames_.size()].data(), entry.frame_id,
                     timestamp_ns, kStrideBytes, {}};
    return true;
  }

  const std::string &timestampMode() const {
    static const std::string scheduled = "scheduled_monotonic";
    static const std::string manifest = "manifest_relative_rebased_to_monotonic";
    return manifest_path_.empty() ? scheduled : manifest;
  }

  const std::string &manifestPath() const { return manifest_path_; }

 private:
  struct ReplayEntry {
    std::string path;
    uint64_t frame_id = 0;
    uint64_t timestamp_ns = 0;
  };

  void loadEntries() {
    const uint64_t prepare_start = monotonicNs();
    for (const auto &entry : entries_) {
      const auto &path = entry.path;
      std::ifstream input(path, std::ios::binary | std::ios::ate);
      if (!input) throw std::runtime_error("Cannot open replay frame: " + path);
      const auto size = input.tellg();
      if (size != static_cast<std::streamoff>(kFrameBytes)) {
        throw std::runtime_error("Replay frame must be packed 1280x720 RGB565 (1843200 bytes): " +
                                 path);
      }
      std::vector<uint16_t> pixels(kFrameBytes / sizeof(uint16_t));
      input.seekg(0);
      input.read(reinterpret_cast<char *>(pixels.data()), static_cast<std::streamsize>(kFrameBytes));
      if (!input) throw std::runtime_error("Short read from replay frame: " + path);
      frames_.push_back(std::move(pixels));
    }
    prepare_ms_ = elapsedMs(prepare_start, monotonicNs());
  }

  const std::string &name() const override {
    static const std::string value = "LOCAL_REPLAY";
    return value;
  }

  const uint16_t *pixelsFor(uint64_t frame_id) const override {
    return frames_[frame_id % frames_.size()].data();
  }

  double prepareMs() const override { return prepare_ms_; }

  std::vector<ReplayEntry> entries_;
  std::vector<std::vector<uint16_t>> frames_;
  std::string manifest_path_;
  double prepare_ms_ = 0.0;
};

class PgvLiveFrameSource final : public FrameSource {
 public:
  explicit PgvLiveFrameSource(std::string path) : path_(std::move(path)) {
    black_.assign(kFrameBytes / sizeof(uint16_t), 0);
    buffers_.reserve(4);
    for (int i = 0; i < 4; ++i) {
      buffers_.push_back(std::make_shared<std::vector<uint16_t>>(kFrameBytes / sizeof(uint16_t)));
    }
  }

  const std::string &name() const override {
    static const std::string value = "PGV_LIVE";
    return value;
  }

  const uint16_t *pixelsFor(uint64_t frame_id) const override {
    (void)frame_id;
    return black_.data();
  }

  bool acquire(uint64_t source_index, uint64_t scheduled_ns, FrameView &view) override {
    (void)source_index;
    (void)scheduled_ns;
    struct stat info {};
    if (stat(path_.c_str(), &info) != 0 || info.st_size != static_cast<off_t>(kPgvFileBytes)) {
      unavailable_samples_++;
      return false;
    }
    const Signature signature{info.st_dev, info.st_ino, info.st_mtime, info.st_mtim.tv_nsec,
                              static_cast<uint64_t>(info.st_size)};
    if (has_signature_ && signature == signature_) return false;

    std::ifstream input(path_, std::ios::binary);
    if (!input) {
      unavailable_samples_++;
      return false;
    }
    std::vector<unsigned char> raw(kPgvFileBytes);
    input.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (input.gcount() != static_cast<std::streamsize>(raw.size())) {
      invalid_frames_++;
      return false;
    }
    const uint32_t version = readU32(raw.data() + 4);
    const uint32_t width = readU32(raw.data() + 8);
    const uint32_t height = readU32(raw.data() + 12);
    const uint32_t frame_id = readU32(raw.data() + 16);
    const uint32_t generation = readU32(raw.data() + 20);
    const uint32_t expected_crc = readU32(raw.data() + 24);
    const uint32_t payload_bytes = readU32(raw.data() + 28);
    uint64_t timestamp_ns = 0;
    std::memcpy(&timestamp_ns, raw.data() + 32, sizeof(timestamp_ns));
    if (std::memcmp(raw.data(), "PGVF", 4) != 0 || version != 1 || width != kWidth ||
        height != kHeight || payload_bytes != kFrameBytes || timestamp_ns == 0 ||
        crc32(raw.data() + kPgvHeaderBytes, kFrameBytes) != expected_crc) {
      invalid_frames_++;
      return false;
    }
    if (has_source_frame_ && generation == generation_) {
      const uint32_t delta = frame_id - last_frame_id_;
      if (delta == 0 || delta >= 0x80000000U) {
        signature_ = signature;
        has_signature_ = true;
        duplicate_frames_++;
        return false;
      }
      if (delta > 1) source_frames_skipped_ += delta - 1;
    }
    auto slot = std::find_if(buffers_.begin(), buffers_.end(),
                             [](const auto &buffer) { return buffer.use_count() == 1; });
    if (slot == buffers_.end()) {
      buffer_busy_drops_++;
      return false;
    }
    std::memcpy((*slot)->data(), raw.data() + kPgvHeaderBytes, kFrameBytes);
    signature_ = signature;
    has_signature_ = true;
    has_source_frame_ = true;
    generation_ = generation;
    last_frame_id_ = frame_id;
    view.owner = *slot;
    view.pixels = view.owner->data();
    view.frame_id = frame_id;
    view.timestamp_ns = timestamp_ns;
    view.stride_bytes = kStrideBytes;
    return true;
  }

  uint64_t sourceFramesSkipped() const { return source_frames_skipped_; }
  uint64_t duplicateFrames() const { return duplicate_frames_; }
  uint64_t invalidFrames() const { return invalid_frames_; }
  uint64_t bufferBusyDrops() const { return buffer_busy_drops_; }
  uint64_t unavailableSamples() const { return unavailable_samples_; }
  const std::string &path() const { return path_; }

 private:
  struct Signature {
    dev_t device;
    ino_t inode;
    time_t seconds;
    long nanoseconds;
    uint64_t bytes;

    bool operator==(const Signature &other) const {
      return device == other.device && inode == other.inode && seconds == other.seconds &&
             nanoseconds == other.nanoseconds && bytes == other.bytes;
    }
  };

  static constexpr size_t kPgvHeaderBytes = 40;
  static constexpr size_t kPgvFileBytes = kPgvHeaderBytes + kFrameBytes;

  static uint32_t readU32(const unsigned char *bytes) {
    uint32_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
  }

  static uint32_t crc32(const unsigned char *bytes, size_t length) {
    static const std::array<uint32_t, 256> table = []() {
      std::array<uint32_t, 256> values{};
      for (uint32_t index = 0; index < values.size(); ++index) {
        uint32_t value = index;
        for (int bit = 0; bit < 8; ++bit) {
          value = (value & 1U) ? (value >> 1U) ^ 0xedb88320U : value >> 1U;
        }
        values[index] = value;
      }
      return values;
    }();
    uint32_t crc = 0xffffffffU;
    for (size_t index = 0; index < length; ++index) {
      crc = table[(crc ^ bytes[index]) & 0xffU] ^ (crc >> 8U);
    }
    return ~crc;
  }

  std::string path_;
  std::vector<uint16_t> black_;
  std::vector<std::shared_ptr<std::vector<uint16_t>>> buffers_;
  Signature signature_{};
  bool has_signature_ = false;
  bool has_source_frame_ = false;
  uint32_t generation_ = 0;
  uint32_t last_frame_id_ = 0;
  uint64_t source_frames_skipped_ = 0;
  uint64_t duplicate_frames_ = 0;
  uint64_t invalid_frames_ = 0;
  uint64_t buffer_busy_drops_ = 0;
  uint64_t unavailable_samples_ = 0;
};

class SyntheticFrameSource final : public FrameSource {
 public:
  SyntheticFrameSource() {
    const uint64_t prepare_start = monotonicNs();
    background_.resize(kFrameBytes / sizeof(uint16_t));
    for (uint32_t y = 0; y < kHeight; ++y) {
      const bool sky = y < 264U;
      const uint16_t color = sky
          ? rgb565(94U + y / 12U, 158U + y / 16U, 191U + y / 24U)
          : rgb565(60U + std::min(34U, (y - 264U) / 14U),
                   67U + std::min(32U, (y - 264U) / 14U),
                   71U + std::min(27U, (y - 264U) / 15U));
      std::fill(background_.begin() + static_cast<size_t>(y) * kWidth,
                background_.begin() + static_cast<size_t>(y + 1U) * kWidth, color);
      if (y >= 264U) {
        const float depth = static_cast<float>(y - 264U);
        const int left = std::max(0, 640 - static_cast<int>(depth * 0.76f));
        const int right = std::min(static_cast<int>(kWidth), 640 + static_cast<int>(depth * 0.76f));
        const uint16_t road = rgb565(42U + std::min(22U, (y - 264U) / 22U),
                                     47U + std::min(21U, (y - 264U) / 22U),
                                     52U + std::min(18U, (y - 264U) / 24U));
        std::fill(background_.begin() + static_cast<size_t>(y) * kWidth + left,
                  background_.begin() + static_cast<size_t>(y) * kWidth + right, road);
      }
    }
    for (int y = 264; y < static_cast<int>(kHeight); ++y) {
      const float depth = static_cast<float>(y - 264);
      const int left = std::max(0, 640 - static_cast<int>(depth * 0.76f));
      const int right = std::min(static_cast<int>(kWidth) - 1, 640 + static_cast<int>(depth * 0.76f));
      if (left > 2) fillRect(background_, left - 3, y, 3, 1, rgb565(229, 222, 179));
      if (right + 3 < static_cast<int>(kWidth)) fillRect(background_, right, y, 3, 1, rgb565(229, 222, 179));
    }
    buffers_.reserve(4);
    for (int i = 0; i < 4; ++i) {
      buffers_.push_back(std::make_shared<std::vector<uint16_t>>(kFrameBytes / sizeof(uint16_t)));
    }
    generate(*buffers_[0], 0);
    prepare_ms_ = elapsedMs(prepare_start, monotonicNs());
  }

  const std::string &name() const override {
    static const std::string value = "SYNTHETIC";
    return value;
  }

  const uint16_t *pixelsFor(uint64_t frame_id) const override {
    (void)frame_id;
    return buffers_[0]->data();
  }

  bool acquire(uint64_t source_index, uint64_t scheduled_ns, FrameView &view) override {
    auto slot = std::find_if(buffers_.begin(), buffers_.end(),
                             [](const auto &buffer) { return buffer.use_count() == 1; });
    if (slot == buffers_.end()) return false;
    generate(**slot, source_index);
    view.owner = *slot;
    view.pixels = view.owner->data();
    view.frame_id = source_index;
    view.timestamp_ns = scheduled_ns;
    view.stride_bytes = kStrideBytes;
    return true;
  }

  double prepareMs() const override { return prepare_ms_; }

 private:
  static uint16_t rgb565(uint8_t r, uint8_t g, uint8_t b) {
    return static_cast<uint16_t>(((r >> 3U) << 11U) | ((g >> 2U) << 5U) | (b >> 3U));
  }

  static void fillRect(std::vector<uint16_t> &frame, int x, int y, int width, int height,
                       uint16_t color) {
    const int left = std::clamp(x, 0, static_cast<int>(kWidth));
    const int top = std::clamp(y, 0, static_cast<int>(kHeight));
    const int right = std::clamp(x + width, 0, static_cast<int>(kWidth));
    const int bottom = std::clamp(y + height, 0, static_cast<int>(kHeight));
    for (int row = top; row < bottom; ++row) {
      std::fill(frame.begin() + static_cast<size_t>(row) * kWidth + left,
                frame.begin() + static_cast<size_t>(row) * kWidth + right, color);
    }
  }

  void generate(std::vector<uint16_t> &frame, uint64_t frame_id) const {
    std::copy(background_.begin(), background_.end(), frame.begin());
    const uint32_t phase = static_cast<uint32_t>((frame_id * 7U) % 420U);
    for (uint32_t segment = 0; segment < 5; ++segment) {
      const int y = 276 + static_cast<int>((phase + segment * 92U) % 430U);
      const int width = std::max(2, (y - 264) / 15);
      const int height = std::max(5, (y - 264) / 11);
      const int center_x = 640;
      fillRect(frame, center_x - width - width / 2, y, width, height,
               rgb565(220, 219, 186));
      fillRect(frame, center_x + width / 2, y, width, height,
               rgb565(220, 219, 186));
    }
    const int car_y = 376 + static_cast<int>((frame_id * 3U) % 285U);
    const int car_x = 812 + static_cast<int>((frame_id * 5U) % 175U);
    const int car_width = std::max(24, (car_y - 250) / 5);
    const int car_height = std::max(16, car_width * 2 / 3);
    fillRect(frame, car_x, car_y, car_width, car_height, rgb565(91, 127, 151));
    fillRect(frame, car_x + car_width / 5, car_y + car_height / 6,
             car_width * 3 / 5, car_height / 3, rgb565(137, 180, 191));
    fillRect(frame, car_x - 2, car_y + car_height * 3 / 4,
             car_width + 4, std::max(2, car_height / 6), rgb565(27, 30, 33));

    // A visible binary source-ID marker guarantees unique synthetic pixels without image synthesis on EGL.
    for (uint32_t bit = 0; bit < 24; ++bit) {
      const bool set = ((frame_id >> bit) & 1U) != 0;
      fillRect(frame, 18 + static_cast<int>(bit * 5U), 20, 4, 12,
               set ? rgb565(246, 246, 232) : rgb565(27, 36, 42));
    }
    fillRect(frame, 44, 356, 9, 9, 0xf800);
  }

  std::vector<uint16_t> background_;
  std::vector<std::shared_ptr<std::vector<uint16_t>>> buffers_;
  double prepare_ms_ = 0.0;
};

class StaticHmiLayer {
 public:
  StaticHmiLayer(const std::string &source, const std::string &mode, uint32_t focus_quadrant,
                 uint32_t navigation_quadrant) {
    const uint64_t started = monotonicNs();
    (void)navigation_quadrant;
    pixels_.assign(kWidth * kHeight * 4U, 0);
    // One cached texture keeps the road dominant and avoids a full-frame CPU redraw.
    fill(0, 0, kWidth, 62, 7, 15, 22, 232);
    fill(0, 61, kWidth, 1, 57, 180, 177, 180);
    fill(24, 16, 8, 22, 101, 214, 207, 240);
    text("ADAS", 46, 18, 3, 235, 244, 245, 255);
    text("MALI EGL", 132, 23, 2, 143, 166, 178, 230);
    text(source.c_str(), 308, 24, 2, 143, 166, 178, 230);
    const uint32_t mode_x[] = {488, 584, 680};
    const char *mode_text[] = {"FOCUS", "MULTI", "SPACE"};
    const bool focus_mode = mode == "FOCUS";
    const bool multi_mode = mode == "MULTI";
    const bool spatial_mode = mode == "SPATIAL";
    const bool selected[] = {focus_mode, multi_mode, spatial_mode};
    for (size_t index = 0; index < 3; ++index) {
      fill(mode_x[index], 16, 84, 30, selected[index] ? 26 : 17, selected[index] ? 73 : 31,
           selected[index] ? 92 : 39, selected[index] ? 240 : 210);
      border(mode_x[index], 16, 84, 30, 1, selected[index] ? 73 : 52,
             selected[index] ? 180 : 77, selected[index] ? 215 : 88, 220);
      text(mode_text[index], mode_x[index] + 14, 26, 2,
           selected[index] ? 151 : 184, selected[index] ? 225 : 194,
           selected[index] ? 232 : 201, 245);
    }
    text("VIEW  CAM 1", 778, 24, 2, 143, 166, 178, 230);
    text("PCIE", 928, 23, 2, 143, 166, 178, 230);
    fill(978, 25, 8, 8, 255, 190, 100, 245);
    text("UNKNOWN", 998, 23, 2, 255, 190, 100, 245);
    text("VIDEO  -- FPS", 1128, 23, 2, 143, 166, 178, 230);

    // Reference layout: small perception card at upper-left, right dynamic stack, and a
    // persistent bottom strip containing four camera previews plus the spatial inset.
    fill(24, 78, 300, 182, 7, 15, 22, 204);
    border(24, 78, 300, 182, 1, 61, 174, 170, 220);
    fill(42, 105, 3, 88, 101, 214, 207, 225);
    text("SIGNAL / PERCEPTION", 64, 92, 2, 143, 166, 178, 235);
    text("UNKNOWN", 64, 120, 3, 255, 190, 100, 250);
    text("VEHICLE   PERSON   OTHER", 48, 175, 2, 143, 166, 178, 230);
    text("--        --        --", 58, 200, 3, 214, 225, 230, 245);
    text("LANE / RULE   UNKNOWN", 48, 236, 2, 143, 166, 178, 220);

    fill(1020, 78, 238, 136, 7, 15, 22, 204);
    border(1020, 78, 238, 136, 1, 61, 174, 170, 220);
    text("NAVIGATION", 1038, 94, 2, 143, 166, 178, 235);
    text("UNREADABLE", 1038, 124, 3, 255, 190, 100, 250);
    text("ROAD NAME UNKNOWN", 1038, 166, 2, 143, 166, 178, 220);
    text("OCR / TTL 750 MS", 1038, 188, 2, 143, 166, 178, 220);
    fill(1020, 224, 238, 148, 7, 15, 22, 204);
    border(1020, 224, 238, 148, 1, 137, 52, 67, 230);
    text("HAZARDS", 1038, 241, 2, 255, 132, 135, 245);
    text("NO VALID EVENT", 1038, 276, 2, 143, 166, 178, 220);
    text("TTL / SOURCE CHECK", 1038, 316, 2, 143, 166, 178, 220);
    fill(1020, 382, 238, 148, 7, 15, 22, 204);
    border(1020, 382, 238, 148, 1, 61, 140, 207, 220);
    text("LANE STATUS", 1038, 399, 2, 143, 196, 231, 245);
    text("CURRENT  UNKNOWN", 1038, 436, 2, 143, 166, 178, 220);
    text("LEFT / RIGHT  UNKNOWN", 1038, 472, 2, 143, 166, 178, 220);
    text("FPGA METADATA  --", 1038, 508, 2, 143, 166, 178, 220);

    const uint32_t preview_x[] = {24, 250, 476, 702};
    const char *preview_text[] = {"CAM 1 / ROAD", "CAM 2 / ROAD", "CAM 3 / ROAD", "CAM 4 / NAV"};
    for (size_t index = 0; index < 4; ++index) {
      fill(preview_x[index], 548, 214, 148, 7, 15, 22, 216);
      border(preview_x[index], 548, 214, 148, 1, 44, 105, 126, 220);
      text(preview_text[index], preview_x[index] + 12, 674, 2, 214, 225, 230, 240);
    }
    fill(928, 548, 326, 148, 7, 15, 22, 224);
    border(928, 548, 326, 148, 1, 61, 174, 170, 220);
    text("ADAS SPACE / SCHEMATIC", 950, 568, 2, 214, 225, 230, 245);
    text("CALIBRATION UNKNOWN", 950, 680, 2, 255, 190, 100, 230);

    fill(24, 700, 1230, 20, 7, 15, 22, 232);
    text("VIEWS", 44, 705, 2, 143, 166, 178, 230);
    const uint8_t active_r = 101, active_g = 214, active_b = 207;
    const uint32_t button_x[] = {150, 294, 438, 582, 726, 870};
    const uint32_t button_w[] = {126, 126, 126, 126, 126, 188};
    const char *button_text[] = {"QUAD", "CAM 1", "CAM 2", "CAM 3", "CAM 4", "SPATIAL"};
    for (size_t index = 0; index < 6; ++index) {
      const bool multi = index == 0 && mode == "MULTI";
      fill(button_x[index], 700, button_w[index], 18, 17, 31, 39, multi ? 238 : 210);
      border(button_x[index], 700, button_w[index], 18, 1,
             multi ? active_r : 52, multi ? active_g : 77, multi ? active_b : 88, 220);
      text(button_text[index], button_x[index] + 15, 705, 2,
           multi ? active_r : 214, multi ? active_g : 225, multi ? active_b : 230, 245);
    }
    const size_t focus_index = 1U + std::min<uint32_t>(focus_quadrant, 3U);
    fill(button_x[focus_index], 700, button_w[focus_index], 2,
         active_r, active_g, active_b, 250);
    prepare_ms_ = elapsedMs(started, monotonicNs());
  }

  const std::vector<unsigned char> &pixels() const { return pixels_; }
  double prepareMs() const { return prepare_ms_; }

 private:
  void fill(uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t r, uint8_t g,
            uint8_t b, uint8_t a) {
    const uint32_t x_end = std::min(kWidth, x + width);
    const uint32_t y_end = std::min(kHeight, y + height);
    for (uint32_t row = y; row < y_end; ++row) {
      unsigned char *pixel = pixels_.data() + (static_cast<size_t>(row) * kWidth + x) * 4U;
      for (uint32_t col = x; col < x_end; ++col, pixel += 4) {
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
        pixel[3] = a;
      }
    }
  }

  void border(uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint32_t thickness,
              uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    fill(x, y, width, thickness, r, g, b, a);
    fill(x, y + height - thickness, width, thickness, r, g, b, a);
    fill(x, y, thickness, height, r, g, b, a);
    fill(x + width - thickness, y, thickness, height, r, g, b, a);
  }

 public:
  static std::array<uint8_t, 7> glyph(char value) {
    switch (value) {
      case 'A': return {0x0e, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11};
      case 'B': return {0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e};
      case 'C': return {0x0e, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0e};
      case 'D': return {0x1c, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1c};
      case 'E': return {0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f};
      case 'F': return {0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10};
      case 'G': return {0x0e, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0f};
      case 'H': return {0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11};
      case 'I': return {0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1f};
      case 'J': return {0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0c};
      case 'K': return {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11};
      case 'L': return {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f};
      case 'M': return {0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11};
      case 'N': return {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11};
      case 'O': return {0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e};
      case 'P': return {0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10};
      case 'Q': return {0x0e, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0d};
      case 'R': return {0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11};
      case 'S': return {0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e};
      case 'T': return {0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04};
      case 'U': return {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e};
      case 'V': return {0x11, 0x11, 0x11, 0x11, 0x11, 0x0a, 0x04};
      case 'W': return {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0a};
      case 'X': return {0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11};
      case 'Y': return {0x11, 0x11, 0x0a, 0x04, 0x04, 0x04, 0x04};
      case 'Z': return {0x1f, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1f};
      case '0': return {0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e};
      case '1': return {0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e};
      case '2': return {0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f};
      case '3': return {0x1e, 0x01, 0x01, 0x0e, 0x01, 0x01, 0x1e};
      case '4': return {0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02};
      case '5': return {0x1f, 0x10, 0x10, 0x1e, 0x01, 0x01, 0x1e};
      case '6': return {0x0e, 0x10, 0x10, 0x1e, 0x11, 0x11, 0x0e};
      case '7': return {0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08};
      case '8': return {0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e};
      case '9': return {0x0e, 0x11, 0x11, 0x0f, 0x01, 0x01, 0x0e};
      case '%': return {0x19, 0x19, 0x02, 0x04, 0x08, 0x13, 0x13};
      case '.': return {0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c};
      case ' ': return {0, 0, 0, 0, 0, 0, 0};
      default: return {};
    }
  }

 private:
  void text(const char *value, uint32_t x, uint32_t y, uint32_t scale, uint8_t r, uint8_t g,
            uint8_t b, uint8_t a) {
    for (const char *cursor = value; *cursor; ++cursor) {
      const auto rows = glyph(*cursor);
      for (uint32_t row = 0; row < rows.size(); ++row) {
        for (uint32_t col = 0; col < 5; ++col) {
          if (rows[row] & (1U << (4U - col))) {
            fill(x + col * scale, y + row * scale, scale, scale, r, g, b, a);
          }
        }
      }
      x += 7 * scale;
    }
  }

  std::vector<unsigned char> pixels_;
  double prepare_ms_ = 0.0;
};

struct DetectionBox {
  float x1 = 0.f;
  float y1 = 0.f;
  float x2 = 0.f;
  float y2 = 0.f;
  float r = 0.f;
  float g = 1.f;
  float b = 0.f;
  uint32_t class_id = 255;
  float confidence = 0.f;
};

struct DetectionPacket {
  uint64_t frame_id = 0;
  uint64_t timestamp_ns = 0;
  uint32_t flags = 0;
  uint8_t nav_direction = 0;
  float nav_distance_m = -1.f;
  std::array<DetectionBox, kMaxDetectionBoxes> boxes{};
  size_t count = 0;
};

constexpr uint32_t kFlagRed = 1U << 0;
constexpr uint32_t kFlagGreen = 1U << 1;

DetectionPacket makeDesignPreviewPacket(uint64_t frame_id, uint64_t timestamp_ns) {
  DetectionPacket packet;
  packet.frame_id = frame_id;
  packet.timestamp_ns = timestamp_ns;
  packet.flags = kFlagGreen;
  packet.nav_direction = 2;  // RIGHT in the isolated visual fixture.
  packet.nav_distance_m = 50.f;
  packet.count = 4;
  packet.boxes[0] = DetectionBox{430.f, 294.f, 594.f, 508.f, 0.08f, 0.88f, 0.98f, 2, 0.87f};
  packet.boxes[1] = DetectionBox{690.f, 320.f, 770.f, 510.f, 0.18f, 0.92f, 0.62f, 0, 0.79f};
  packet.boxes[2] = DetectionBox{804.f, 244.f, 948.f, 421.f, 0.08f, 0.88f, 0.98f, 2, 0.92f};
  packet.boxes[3] = DetectionBox{588.f, 124.f, 650.f, 184.f, 0.12f, 0.86f, 0.98f, 9, 0.93f};
  return packet;
}

class SyntheticDetectionWorker {
 public:
  void start(FrameSource &source, uint64_t deadline_ns, uint32_t fps, uint32_t duration_seconds) {
    latency_ms_.reserve(static_cast<size_t>(fps) * duration_seconds);
    running_.store(true);
    worker_ = std::thread([this, &source, deadline_ns]() {
      uint64_t last_frame_id = UINT64_MAX;
      while (running_.load()) {
        FrameView frame;
        if (!source.waitLatest(last_frame_id, deadline_ns, frame)) break;
        if (last_frame_id != UINT64_MAX && frame.frame_id > last_frame_id + 1) {
          skipped_source_ids_ += frame.frame_id - last_frame_id - 1;
        }
        last_frame_id = frame.frame_id;
        const uint64_t started = monotonicNs();
        std::this_thread::sleep_for(std::chrono::milliseconds(18));

        DetectionPacket packet;
        packet.frame_id = frame.frame_id;
        packet.timestamp_ns = frame.timestamp_ns;
        packet.count = 2;
        const uint32_t index = static_cast<uint32_t>(frame.frame_id % kSyntheticPoolFrames);
        const uint32_t bar_left = (index * 23U) % (kWidth - 96U);
        packet.boxes[0] = DetectionBox{static_cast<float>(bar_left), 150.f,
                                       static_cast<float>(bar_left + 96U), 590.f,
                                       0.20f, 0.95f, 0.56f};
        packet.boxes[1] = DetectionBox{884.f, 86.f, 1044.f, 222.f, 1.00f, 0.64f, 0.20f};
        {
          std::lock_guard<std::mutex> lock(packet_mutex_);
          latest_ = packet;
        }
        latency_ms_.push_back(elapsedMs(started, monotonicNs()));
        analyzed_frames_.fetch_add(1, std::memory_order_relaxed);
      }
      running_.store(false);
    });
  }

  bool snapshot(uint64_t current_frame_id, uint64_t now_ns, DetectionPacket &packet) const {
    std::lock_guard<std::mutex> lock(packet_mutex_);
    if (latest_.count == 0 || current_frame_id < latest_.frame_id ||
        current_frame_id - latest_.frame_id > 15 || now_ns < latest_.timestamp_ns ||
        now_ns - latest_.timestamp_ns > 250000000ULL) {
      return false;
    }
    packet = latest_;
    return true;
  }

  void stop() {
    running_.store(false);
    if (worker_.joinable()) worker_.join();
  }

  uint64_t analyzedFrames() const { return analyzed_frames_.load(std::memory_order_relaxed); }
  uint64_t skippedSourceIds() const { return skipped_source_ids_; }
  const std::vector<double> &latenciesMs() const { return latency_ms_; }

 private:
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> analyzed_frames_{0};
  uint64_t skipped_source_ids_ = 0;
  mutable std::mutex packet_mutex_;
  DetectionPacket latest_{};
  std::thread worker_;
  std::vector<double> latency_ms_;
};

class EglX11Window {
 public:
  EglX11Window() { initialize(); }
  ~EglX11Window() { cleanup(); }

  const std::string &eglVendor() const { return egl_vendor_; }
  const std::string &eglVersion() const { return egl_version_; }
  const std::string &glRenderer() const { return gl_renderer_; }
  const std::string &glVersion() const { return gl_version_; }

  void upload(const uint16_t *pixels) {
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kWidth, kHeight, GL_RGB,
                    GL_UNSIGNED_SHORT_5_6_5, pixels);
  }

  void draw() {
    glViewport(0, 0, kWidth, kHeight);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(program_);
    glUniform4f(crop_location_, crop_[0], crop_[1], crop_[2], crop_[3]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glUniform1i(sampler_location_, 0);
    glEnableVertexAttribArray(position_location_);
    glEnableVertexAttribArray(texcoord_location_);
    glVertexAttribPointer(position_location_, 2, GL_FLOAT, 0, 4 * sizeof(GLfloat), vertices_);
    glVertexAttribPointer(texcoord_location_, 2, GL_FLOAT, 0, 4 * sizeof(GLfloat),
                          vertices_ + 2);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  }

  void setVideoCrop(uint32_t quadrant, bool crop_front) {
    crop_ = {0.f, 0.f, 1.f, 1.f};
    if (!crop_front) return;
    const uint32_t row = quadrant / 2U;
    const uint32_t column = quadrant % 2U;
    crop_[0] = static_cast<GLfloat>(column) * 0.5f;
    crop_[1] = static_cast<GLfloat>(row) * 0.5f;
    crop_[2] = static_cast<GLfloat>(column + 1U) * 0.5f;
    crop_[3] = static_cast<GLfloat>(row + 1U) * 0.5f;
  }

  // Returns queued keyboard/mouse commands without blocking the EGL thread.
  std::vector<char> pollControls() {
    std::vector<char> commands;
    while (XPending(x_display_) > 0) {
      XEvent event{};
      XNextEvent(x_display_, &event);
      if (event.type == KeyPress) {
        const KeySym key = XLookupKeysym(&event.xkey, 0);
        if (key >= XK_1 && key <= XK_4) commands.push_back(static_cast<char>('1' + key - XK_1));
        else if (key == XK_q || key == XK_Q) commands.push_back('q');
        else if (key == XK_m || key == XK_M) commands.push_back('m');
        else if (key == XK_f || key == XK_F) commands.push_back('f');
        else if (key == XK_s || key == XK_S) commands.push_back('s');
        else if (key == XK_n || key == XK_N) commands.push_back('n');
        else if (key == XK_Escape) commands.push_back('x');
      } else if (event.type == ButtonPress && event.xbutton.y >= 650) {
        const int x = event.xbutton.x;
        if (x >= 150 && x < 276) commands.push_back('q');
        else if (x >= 294 && x < 420) commands.push_back('1');
        else if (x >= 438 && x < 564) commands.push_back('2');
        else if (x >= 582 && x < 708) commands.push_back('3');
        else if (x >= 726 && x < 852) commands.push_back('4');
        else if (x >= 870 && x < 1058) commands.push_back('s');
      }
    }
    return commands;
  }

  void drawSpatialOverlay(const DetectionPacket *packet) {
    std::array<GLfloat, (32U + kMaxDetectionBoxes * 8U) * 6U> vertices{};
    size_t offset = 0;
    const auto point = [&vertices, &offset](float x, float y, float r, float g, float b,
                                            float a) {
      vertices[offset++] = (x / static_cast<float>(kWidth)) * 2.f - 1.f;
      vertices[offset++] = 1.f - (y / static_cast<float>(kHeight)) * 2.f;
      vertices[offset++] = r;
      vertices[offset++] = g;
      vertices[offset++] = b;
      vertices[offset++] = a;
    };
    const auto line = [&point](float x1, float y1, float x2, float y2, float r, float g, float b,
                               float a) {
      point(x1, y1, r, g, b, a);
      point(x2, y2, r, g, b, a);
    };
    // Cached camera-space guide: this is explicitly schematic until calibration is available.
    line(428.f, 650.f, 562.f, 350.f, 0.40f, 0.84f, 0.82f, 0.72f);
    line(852.f, 650.f, 718.f, 350.f, 0.40f, 0.84f, 0.82f, 0.72f);
    line(512.f, 650.f, 610.f, 390.f, 0.28f, 0.52f, 0.55f, 0.58f);
    line(768.f, 650.f, 670.f, 390.f, 0.28f, 0.52f, 0.55f, 0.58f);
    line(604.f, 614.f, 676.f, 614.f, 0.94f, 0.75f, 0.35f, 0.92f);
    line(676.f, 614.f, 676.f, 660.f, 0.94f, 0.75f, 0.35f, 0.92f);
    line(676.f, 660.f, 604.f, 660.f, 0.94f, 0.75f, 0.35f, 0.92f);
    line(604.f, 660.f, 604.f, 614.f, 0.94f, 0.75f, 0.35f, 0.92f);
    if (packet) {
      for (size_t index = 0; index < packet->count && index < packet->boxes.size(); ++index) {
        const DetectionBox &box = packet->boxes[index];
        const float center_x = std::clamp((box.x1 + box.x2) * 0.5f, 0.f, 1279.f);
        const float center_y = std::clamp((box.y1 + box.y2) * 0.5f, 0.f, 719.f);
        const float depth = std::clamp(1.f - center_y / 720.f, 0.05f, 0.95f);
        const float x = 640.f + (center_x - 640.f) * (0.20f + 0.35f * depth);
        const float y = 578.f - depth * 210.f;
        const float width = std::clamp((box.x2 - box.x1) * 0.20f, 10.f, 82.f);
        const float height = std::clamp((box.y2 - box.y1) * 0.15f, 8.f, 50.f);
        line(x - width, y - height, x + width, y - height, box.r, box.g, box.b, 0.92f);
        line(x + width, y - height, x + width, y + height, box.r, box.g, box.b, 0.92f);
        line(x + width, y + height, x - width, y + height, box.r, box.g, box.b, 0.92f);
        line(x - width, y + height, x - width, y - height, box.r, box.g, box.b, 0.92f);
      }
    }
    if (offset == 0) return;
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(box_program_);
    glEnableVertexAttribArray(box_position_location_);
    glEnableVertexAttribArray(box_color_location_);
    glVertexAttribPointer(box_position_location_, 2, GL_FLOAT, 0, 6 * sizeof(GLfloat),
                          vertices.data());
    glVertexAttribPointer(box_color_location_, 4, GL_FLOAT, 0, 6 * sizeof(GLfloat),
                          vertices.data() + 2);
    glLineWidth(1.f);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(offset / 6));
    glDisable(GL_BLEND);
  }

  void setStaticUiPixels(const unsigned char *pixels) {
    if (!ui_texture_) glGenTextures(1, &ui_texture_);
    glBindTexture(GL_TEXTURE_2D, ui_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, kWidth, kHeight, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, pixels);
    if (glGetError() != GL_NO_ERROR) throw std::runtime_error("Static HMI texture upload failed");
  }

  bool loadStaticUiFile(const std::string &path) {
    if (path.empty()) return false;
    struct stat info {};
    if (stat(path.c_str(), &info) != 0) return false;
    constexpr size_t header_bytes = 24;
    constexpr size_t pixel_bytes = static_cast<size_t>(kWidth) * kHeight * 4U;
    if (info.st_size != static_cast<off_t>(header_bytes + pixel_bytes)) return false;
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    std::vector<unsigned char> raw(header_bytes + pixel_bytes);
    input.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (input.gcount() != static_cast<std::streamsize>(raw.size()) ||
        std::memcmp(raw.data(), "UI1\0", 4) != 0) {
      return false;
    }
    uint16_t file_width = 0, file_height = 0;
    uint32_t file_bytes = 0;
    std::memcpy(&file_width, raw.data() + 4, sizeof(file_width));
    std::memcpy(&file_height, raw.data() + 6, sizeof(file_height));
    std::memcpy(&file_bytes, raw.data() + 20, sizeof(file_bytes));
    if (file_width != kWidth || file_height != kHeight || file_bytes != pixel_bytes) return false;
    setStaticUiPixels(raw.data() + header_bytes);
    return true;
  }

  bool loadDynamicUiFile(const std::string &path) {
    if (path.empty()) return false;
    struct stat info {};
    if (stat(path.c_str(), &info) != 0) return false;
    const HmiSignature signature{info.st_dev, info.st_ino, info.st_mtime, info.st_mtim.tv_nsec,
                                 static_cast<uint64_t>(info.st_size)};
    if (has_dynamic_signature_ && signature == dynamic_signature_) return false;
    constexpr uint32_t width = 238;
    constexpr uint32_t height = 452;
    constexpr size_t header_bytes = 24;
    constexpr size_t pixel_bytes = static_cast<size_t>(width) * height * 4U;
    if (info.st_size != static_cast<off_t>(header_bytes + pixel_bytes)) return false;
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    std::vector<unsigned char> raw(header_bytes + pixel_bytes);
    input.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (input.gcount() != static_cast<std::streamsize>(raw.size()) ||
        std::memcmp(raw.data(), "HMI1", 4) != 0) {
      return false;
    }
    uint16_t file_width = 0, file_height = 0;
    std::memcpy(&file_width, raw.data() + 4, sizeof(file_width));
    std::memcpy(&file_height, raw.data() + 6, sizeof(file_height));
    if (file_width != width || file_height != height) return false;
    std::memcpy(&dynamic_hmi_frame_id_, raw.data() + 8, sizeof(dynamic_hmi_frame_id_));
    std::memcpy(&dynamic_hmi_timestamp_ns_, raw.data() + 16, sizeof(dynamic_hmi_timestamp_ns_));
    setDynamicUiPixels(raw.data() + header_bytes);
    dynamic_signature_ = signature;
    has_dynamic_signature_ = true;
    return true;
  }

  void drawDynamicUi() {
    if (!dynamic_ui_texture_) return;
    const uint64_t now_ns = monotonicNs();
    if (now_ns < dynamic_hmi_timestamp_ns_ ||
        now_ns - dynamic_hmi_timestamp_ns_ > 750000000ULL) {
      return;
    }
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    drawTexturedRect(dynamic_ui_texture_, 1020.f, 78.f, 238.f, 452.f);
    glDisable(GL_BLEND);
  }

  void drawVideoThumbnails(bool quad_source) {
    for (size_t index = 0; index < 4; ++index) {
      const uint32_t row = static_cast<uint32_t>(index / 2U);
      const uint32_t column = static_cast<uint32_t>(index % 2U);
      const GLfloat u0 = quad_source ? column * 0.5f : 0.f;
      const GLfloat v0 = quad_source ? row * 0.5f : 0.f;
      const GLfloat u1 = quad_source ? (column + 1U) * 0.5f : 1.f;
      const GLfloat v1 = quad_source ? (row + 1U) * 0.5f : 1.f;
      const UiRect &card = kLayout.thumbnails[index];
      drawTexturedRectCrop(texture_, card.x + 6.f, card.y + 6.f, card.width - 12.f,
                           card.height - 42.f, u0, v0, u1, v1);
    }
  }

  void drawLiveStatus(const DetectionPacket &packet) {
    if (!status_texture_ || status_frame_id_ != packet.frame_id) updateStatusTexture(packet);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    drawTexturedRect(status_texture_, 24.f, 78.f, 300.f, 182.f);
    glDisable(GL_BLEND);
  }

  void drawSpatialInset(const DetectionPacket *packet) {
    std::array<GLfloat, (16U + kMaxDetectionBoxes * 8U) * 6U> vertices{};
    size_t offset = 0;
    const auto point = [&vertices, &offset](float x, float y, float r, float g, float b,
                                            float a) {
      vertices[offset++] = (x / static_cast<float>(kWidth)) * 2.f - 1.f;
      vertices[offset++] = 1.f - (y / static_cast<float>(kHeight)) * 2.f;
      vertices[offset++] = r;
      vertices[offset++] = g;
      vertices[offset++] = b;
      vertices[offset++] = a;
    };
    const auto line = [&point](float x1, float y1, float x2, float y2, float r, float g, float b,
                               float a) {
      point(x1, y1, r, g, b, a);
      point(x2, y2, r, g, b, a);
    };
    line(1038.f, 682.f, 1080.f, 586.f, 0.40f, 0.84f, 0.82f, 0.78f);
    line(1172.f, 586.f, 1214.f, 682.f, 0.40f, 0.84f, 0.82f, 0.78f);
    line(1072.f, 682.f, 1118.f, 606.f, 0.27f, 0.54f, 0.58f, 0.66f);
    line(1180.f, 606.f, 1226.f, 682.f, 0.27f, 0.54f, 0.58f, 0.66f);
    line(1110.f, 662.f, 1160.f, 662.f, 0.95f, 0.76f, 0.35f, 0.95f);
    line(1160.f, 662.f, 1160.f, 688.f, 0.95f, 0.76f, 0.35f, 0.95f);
    line(1160.f, 688.f, 1110.f, 688.f, 0.95f, 0.76f, 0.35f, 0.95f);
    line(1110.f, 688.f, 1110.f, 662.f, 0.95f, 0.76f, 0.35f, 0.95f);
    if (packet) {
      for (size_t index = 0; index < packet->count && index < packet->boxes.size(); ++index) {
        const DetectionBox &box = packet->boxes[index];
        const float cx = std::clamp((box.x1 + box.x2) * 0.5f, 0.f, 1279.f);
        const float cy = std::clamp((box.y1 + box.y2) * 0.5f, 0.f, 719.f);
        const float depth = std::clamp(1.f - cy / 720.f, 0.05f, 0.95f);
        const float x = 1135.f + (cx - 640.f) * (0.055f + 0.08f * depth);
        const float y = 650.f - depth * 42.f;
        const float width = std::clamp((box.x2 - box.x1) * 0.055f, 3.f, 22.f);
        line(x - width, y, x + width, y, box.r, box.g, box.b, 0.94f);
        line(x + width, y, x + width, y + 10.f, box.r, box.g, box.b, 0.94f);
        line(x + width, y + 10.f, x - width, y + 10.f, box.r, box.g, box.b, 0.94f);
        line(x - width, y + 10.f, x - width, y, box.r, box.g, box.b, 0.94f);
      }
    }
    if (offset == 0) return;
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(box_program_);
    glEnableVertexAttribArray(box_position_location_);
    glEnableVertexAttribArray(box_color_location_);
    glVertexAttribPointer(box_position_location_, 2, GL_FLOAT, 0, 6 * sizeof(GLfloat),
                          vertices.data());
    glVertexAttribPointer(box_color_location_, 4, GL_FLOAT, 0, 6 * sizeof(GLfloat),
                          vertices.data() + 2);
    glLineWidth(1.f);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(offset / 6));
    glDisable(GL_BLEND);
  }

  uint64_t dynamicHmiFrameId() const { return dynamic_hmi_frame_id_; }
  uint64_t dynamicHmiTimestampNs() const { return dynamic_hmi_timestamp_ns_; }

  void drawStaticUi() {
    if (!ui_texture_) return;
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    drawTextured(ui_texture_);
    glDisable(GL_BLEND);
  }

  void drawDetectionBoxes(const DetectionPacket &packet) {
    if (packet.count == 0) return;
    std::array<GLfloat, kMaxDetectionBoxes * 48> vertices{};
    size_t offset = 0;
    const auto point = [&vertices, &offset](float x, float y, const DetectionBox &box) {
      vertices[offset++] = (x / static_cast<float>(kWidth)) * 2.f - 1.f;
      vertices[offset++] = 1.f - (y / static_cast<float>(kHeight)) * 2.f;
      vertices[offset++] = box.r;
      vertices[offset++] = box.g;
      vertices[offset++] = box.b;
      vertices[offset++] = 0.96f;
    };
    for (size_t i = 0; i < packet.count && i < packet.boxes.size(); ++i) {
      const DetectionBox &box = packet.boxes[i];
      point(box.x1, box.y1, box); point(box.x2, box.y1, box);
      point(box.x2, box.y1, box); point(box.x2, box.y2, box);
      point(box.x2, box.y2, box); point(box.x1, box.y2, box);
      point(box.x1, box.y2, box); point(box.x1, box.y1, box);
    }
    const GLint scissor_x = static_cast<GLint>(kLayout.main_video.x);
    const GLint scissor_y = static_cast<GLint>(kHeight -
                                               (kLayout.main_video.y + kLayout.main_video.height));
    const GLsizei scissor_width = static_cast<GLsizei>(kLayout.main_video.width);
    const GLsizei scissor_height = static_cast<GLsizei>(kLayout.main_video.height);
    glEnable(GL_SCISSOR_TEST);
    glScissor(scissor_x, scissor_y, scissor_width, scissor_height);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(box_program_);
    glEnableVertexAttribArray(box_position_location_);
    glEnableVertexAttribArray(box_color_location_);
    glVertexAttribPointer(box_position_location_, 2, GL_FLOAT, 0, 6 * sizeof(GLfloat), vertices.data());
    glVertexAttribPointer(box_color_location_, 4, GL_FLOAT, 0, 6 * sizeof(GLfloat),
                          vertices.data() + 2);
    glLineWidth(1.f);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(offset / 6));
    glDisable(GL_BLEND);
    if (!has_label_frame_ || label_frame_id_ != packet.frame_id) updateLabelAtlas(packet);
    drawDetectionLabels(packet);
    glDisable(GL_SCISSOR_TEST);
  }

  void verifySyntheticRedPixel() {
    unsigned char pixel[4] = {};
    glFinish();
    glReadPixels(48, 360, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
    if (glGetError() != GL_NO_ERROR || pixel[0] < 200 || pixel[1] > 40 || pixel[2] > 40) {
      std::ostringstream message;
      message << "RGB565 render readback mismatch: rgba=" << static_cast<int>(pixel[0]) << ','
              << static_cast<int>(pixel[1]) << ',' << static_cast<int>(pixel[2]) << ','
              << static_cast<int>(pixel[3]);
      throw std::runtime_error(message.str());
    }
  }

  void swap() {
    if (!eglSwapBuffers(egl_display_, egl_surface_)) {
      throw std::runtime_error("eglSwapBuffers failed: " + eglError());
    }
  }

 private:
  static constexpr GLfloat vertices_[16] = {
      -1.f, -1.f, 0.f, 1.f, 1.f, -1.f, 1.f, 1.f,
      -1.f, 1.f,  0.f, 0.f, 1.f, 1.f,  1.f, 0.f};

  static void statusText(std::vector<unsigned char> &pixels, uint32_t width, uint32_t height,
                         const char *value, uint32_t x, uint32_t y, uint32_t scale,
                         uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    for (const char *cursor = value; *cursor; ++cursor) {
      const auto rows = StaticHmiLayer::glyph(*cursor);
      for (uint32_t row = 0; row < rows.size(); ++row) {
        for (uint32_t col = 0; col < 5; ++col) {
          if ((rows[row] & (1U << (4U - col))) == 0) continue;
          for (uint32_t dy = 0; dy < scale; ++dy) {
            for (uint32_t dx = 0; dx < scale; ++dx) {
              const uint32_t px = x + col * scale + dx;
              const uint32_t py = y + row * scale + dy;
              if (px >= width || py >= height) continue;
              const size_t index = (static_cast<size_t>(py) * width + px) * 4U;
              pixels[index] = r;
              pixels[index + 1] = g;
              pixels[index + 2] = b;
              pixels[index + 3] = a;
            }
          }
        }
      }
      x += 7U * scale;
    }
  }

  void updateStatusTexture(const DetectionPacket &packet) {
    constexpr uint32_t width = 300, height = 182;
    std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 4U, 0);
    const auto fill = [&pixels, width, height](uint32_t x, uint32_t y, uint32_t w, uint32_t h, uint8_t r,
                                uint8_t g, uint8_t b, uint8_t a) {
      const uint32_t right = std::min<uint32_t>(width, x + w);
      const uint32_t bottom = std::min<uint32_t>(height, y + h);
      for (uint32_t row = y; row < bottom; ++row) {
        for (uint32_t col = x; col < right; ++col) {
          const size_t index = (static_cast<size_t>(row) * width + col) * 4U;
          pixels[index] = r;
          pixels[index + 1] = g;
          pixels[index + 2] = b;
          pixels[index + 3] = a;
        }
      }
    };
    fill(0, 0, width, height, 7, 15, 22, 220);
    fill(0, 0, 2, height, 61, 174, 170, 235);
    const bool red = (packet.flags & kFlagRed) != 0;
    const bool green = (packet.flags & kFlagGreen) != 0;
    const uint8_t r = red ? 255 : (green ? 92 : 255);
    const uint8_t g = red ? 91 : (green ? 220 : 190);
    const uint8_t b = red ? 88 : (green ? 157 : 100);
    const char *signal = red ? "RED" : green ? "GREEN" : "UNKNOWN";
    fill(24, 30, 26, 66, r, g, b, 215);
    statusText(pixels, width, height, "SIGNAL", 70, 14, 2, 143, 166, 178, 245);
    statusText(pixels, width, height, signal, 70, 42, 3, r, g, b, 255);
    uint32_t vehicles = 0, persons = 0, other = 0;
    for (size_t index = 0; index < packet.count && index < packet.boxes.size(); ++index) {
      if (packet.boxes[index].class_id == 0) ++persons;
      else if (packet.boxes[index].class_id == 1 || packet.boxes[index].class_id == 3) ++other;
      else if (packet.boxes[index].class_id == 2 || packet.boxes[index].class_id == 5 ||
               packet.boxes[index].class_id == 7) ++vehicles;
      else ++other;
    }
    statusText(pixels, width, height, "VEHICLE", 24, 120, 2, 143, 166, 178, 230);
    statusText(pixels, width, height, "PERSON", 124, 120, 2, 143, 166, 178, 230);
    statusText(pixels, width, height, "OTHER", 216, 120, 2, 143, 166, 178, 230);
    const std::string counts = std::to_string(vehicles) + "        " + std::to_string(persons) +
                               "        " + std::to_string(other);
    statusText(pixels, width, height, counts.c_str(), 34, 146, 3, 214, 225, 230, 245);
    statusText(pixels, width, height, "LANE / RULE   UNKNOWN", 24, 174, 2, 143, 166, 178, 220);
    if (!status_texture_) {
      glGenTextures(1, &status_texture_);
      glBindTexture(GL_TEXTURE_2D, status_texture_);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, pixels.data());
    } else {
      glBindTexture(GL_TEXTURE_2D, status_texture_);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA,
                      GL_UNSIGNED_BYTE, pixels.data());
    }
    status_frame_id_ = packet.frame_id;
  }

  void drawTextured(GLuint texture) {
    glViewport(0, 0, kWidth, kHeight);
    glUseProgram(program_);
    glUniform4f(crop_location_, 0.f, 0.f, 1.f, 1.f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glUniform1i(sampler_location_, 0);
    glEnableVertexAttribArray(position_location_);
    glEnableVertexAttribArray(texcoord_location_);
    glVertexAttribPointer(position_location_, 2, GL_FLOAT, 0, 4 * sizeof(GLfloat), vertices_);
    glVertexAttribPointer(texcoord_location_, 2, GL_FLOAT, 0, 4 * sizeof(GLfloat),
                          vertices_ + 2);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  }

  void drawTexturedRect(GLuint texture, GLfloat x, GLfloat y, GLfloat width, GLfloat height) {
    drawTexturedRectCrop(texture, x, y, width, height, 0.f, 0.f, 1.f, 1.f);
  }

  void drawTexturedRectCrop(GLuint texture, GLfloat x, GLfloat y, GLfloat width, GLfloat height,
                            GLfloat u0, GLfloat v0, GLfloat u1, GLfloat v1) {
    const GLfloat left = -1.f + 2.f * x / static_cast<float>(kWidth);
    const GLfloat right = -1.f + 2.f * (x + width) / static_cast<float>(kWidth);
    const GLfloat top = 1.f - 2.f * y / static_cast<float>(kHeight);
    const GLfloat bottom = 1.f - 2.f * (y + height) / static_cast<float>(kHeight);
    const GLfloat vertices[] = {
        left, bottom, u0, v1, right, bottom, u1, v1,
        left, top,    u0, v0, right, top,    u1, v0};
    glViewport(0, 0, kWidth, kHeight);
    glUseProgram(program_);
    glUniform4f(crop_location_, 0.f, 0.f, 1.f, 1.f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glUniform1i(sampler_location_, 0);
    glEnableVertexAttribArray(position_location_);
    glEnableVertexAttribArray(texcoord_location_);
    glVertexAttribPointer(position_location_, 2, GL_FLOAT, 0, 4 * sizeof(GLfloat), vertices);
    glVertexAttribPointer(texcoord_location_, 2, GL_FLOAT, 0, 4 * sizeof(GLfloat), vertices + 2);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  }

  static std::string detectionName(uint32_t class_id) {
    switch (class_id) {
      case 0: return "PERSON";
      case 1: return "BICYCLE";
      case 2: return "CAR";
      case 3: return "MOTORCYCLE";
      case 5: return "BUS";
      case 7: return "TRUCK";
      case 9: return "SIGNAL";
      default: return "OBJECT";
    }
  }

  static std::string detectionLabel(const DetectionBox &box) {
    std::ostringstream text;
    text << detectionName(box.class_id) << ' '
         << static_cast<unsigned>(std::round(box.confidence * 100.f)) << '%';
    return text.str();
  }

  void updateLabelAtlas(const DetectionPacket &packet) {
    constexpr uint32_t cell_width = 180;
    constexpr uint32_t cell_height = 24;
    constexpr uint32_t columns = 4;
    constexpr uint32_t atlas_width = cell_width * columns;
    constexpr uint32_t atlas_height = cell_height * 8;
    std::vector<unsigned char> pixels(static_cast<size_t>(atlas_width) * atlas_height * 4U, 0);
    for (size_t index = 0; index < packet.count && index < packet.boxes.size(); ++index) {
      const DetectionBox &box = packet.boxes[index];
      const uint32_t origin_x = static_cast<uint32_t>(index % columns) * cell_width;
      const uint32_t origin_y = static_cast<uint32_t>(index / columns) * cell_height;
      const unsigned char r = static_cast<unsigned char>(std::clamp(box.r * 75.f, 20.f, 180.f));
      const unsigned char g = static_cast<unsigned char>(std::clamp(box.g * 75.f, 20.f, 180.f));
      const unsigned char b = static_cast<unsigned char>(std::clamp(box.b * 75.f, 20.f, 180.f));
      for (uint32_t y = 0; y < cell_height; ++y) {
        for (uint32_t x = 0; x < cell_width; ++x) {
          const size_t pixel = (static_cast<size_t>(origin_y + y) * atlas_width + origin_x + x) * 4U;
          pixels[pixel] = r;
          pixels[pixel + 1] = g;
          pixels[pixel + 2] = b;
          pixels[pixel + 3] = 224;
        }
      }
      const std::string label = detectionLabel(box).substr(0, 14);
      uint32_t text_x = origin_x + 4;
      for (char character : label) {
        const auto rows = StaticHmiLayer::glyph(character);
        for (uint32_t row = 0; row < rows.size(); ++row) {
          for (uint32_t column = 0; column < 5; ++column) {
            if ((rows[row] & (1U << (4U - column))) == 0) continue;
            for (uint32_t dy = 0; dy < 2; ++dy) {
              for (uint32_t dx = 0; dx < 2; ++dx) {
                const uint32_t x = text_x + column * 2U + dx;
                const uint32_t y = origin_y + 4U + row * 2U + dy;
                const size_t pixel = (static_cast<size_t>(y) * atlas_width + x) * 4U;
                pixels[pixel] = 246;
                pixels[pixel + 1] = 248;
                pixels[pixel + 2] = 244;
                pixels[pixel + 3] = 255;
              }
            }
          }
        }
        text_x += 12;
      }
    }
    if (!label_texture_) {
      glGenTextures(1, &label_texture_);
      glBindTexture(GL_TEXTURE_2D, label_texture_);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, atlas_width, atlas_height, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, pixels.data());
    } else {
      glBindTexture(GL_TEXTURE_2D, label_texture_);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, atlas_width, atlas_height, GL_RGBA,
                      GL_UNSIGNED_BYTE, pixels.data());
    }
    label_frame_id_ = packet.frame_id;
    has_label_frame_ = true;
  }

  void drawDetectionLabels(const DetectionPacket &packet) {
    constexpr GLfloat cell_width = 180.f;
    constexpr GLfloat cell_height = 24.f;
    constexpr GLfloat atlas_width = cell_width * 4.f;
    constexpr GLfloat atlas_height = cell_height * 8.f;
    std::vector<GLfloat> vertices;
    vertices.reserve(packet.count * 6U * 4U);
    const auto vertex = [&vertices](GLfloat x, GLfloat y, GLfloat u, GLfloat v) {
      vertices.push_back(x / static_cast<GLfloat>(kWidth) * 2.f - 1.f);
      vertices.push_back(1.f - y / static_cast<GLfloat>(kHeight) * 2.f);
      vertices.push_back(u);
      vertices.push_back(v);
    };
    for (size_t index = 0; index < packet.count && index < packet.boxes.size(); ++index) {
      const DetectionBox &box = packet.boxes[index];
      const GLfloat left = std::clamp(box.x1, 0.f, static_cast<float>(kWidth - 1));
      const GLfloat top = std::clamp(box.y1 - cell_height, 0.f, static_cast<float>(kHeight - 1));
      const GLfloat right = std::min(static_cast<float>(kWidth), left + cell_width);
      const GLfloat bottom = std::min(static_cast<float>(kHeight), top + cell_height);
      const GLfloat u0 = static_cast<GLfloat>(index % 4U) * cell_width / atlas_width;
      const GLfloat u1 = u0 + cell_width / atlas_width;
      const GLfloat v0 = static_cast<GLfloat>(index / 4U) * cell_height / atlas_height;
      const GLfloat v1 = v0 + cell_height / atlas_height;
      vertex(left, top, u0, v0); vertex(left, bottom, u0, v1); vertex(right, bottom, u1, v1);
      vertex(left, top, u0, v0); vertex(right, bottom, u1, v1); vertex(right, top, u1, v0);
    }
    if (vertices.empty()) return;
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(program_);
    glUniform4f(crop_location_, 0.f, 0.f, 1.f, 1.f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, label_texture_);
    glUniform1i(sampler_location_, 0);
    glEnableVertexAttribArray(position_location_);
    glEnableVertexAttribArray(texcoord_location_);
    glVertexAttribPointer(position_location_, 2, GL_FLOAT, 0, 4 * sizeof(GLfloat), vertices.data());
    glVertexAttribPointer(texcoord_location_, 2, GL_FLOAT, 0, 4 * sizeof(GLfloat), vertices.data() + 2);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size() / 4U));
    glDisable(GL_BLEND);
  }

  void setDynamicUiPixels(const unsigned char *pixels) {
    if (!dynamic_ui_texture_) {
      glGenTextures(1, &dynamic_ui_texture_);
      glBindTexture(GL_TEXTURE_2D, dynamic_ui_texture_);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 238, 452, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, pixels);
    } else {
      glBindTexture(GL_TEXTURE_2D, dynamic_ui_texture_);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 238, 452, GL_RGBA,
                      GL_UNSIGNED_BYTE, pixels);
    }
    if (glGetError() != GL_NO_ERROR) throw std::runtime_error("Dynamic HMI texture upload failed");
  }

  std::string eglError() const {
    std::ostringstream out;
    out << "0x" << std::hex << eglGetError();
    return out.str();
  }

  static std::string glString(GLenum name) {
    const auto *value = glGetString(name);
    return value ? reinterpret_cast<const char *>(value) : "unavailable";
  }

  static GLuint compileShader(GLenum type, const char *source) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
      GLint length = 0;
      glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
      std::vector<char> log(static_cast<size_t>(std::max(1, length)));
      glGetShaderInfoLog(shader, length, nullptr, log.data());
      glDeleteShader(shader);
      throw std::runtime_error(std::string("GL shader compile failed: ") + log.data());
    }
    return shader;
  }

  void initialize() {
    x_display_ = XOpenDisplay(nullptr);
    if (!x_display_) throw std::runtime_error("XOpenDisplay failed; check DISPLAY and XAUTHORITY");
    if (XDisplayWidth(x_display_, DefaultScreen(x_display_)) != static_cast<int>(kWidth) ||
        XDisplayHeight(x_display_, DefaultScreen(x_display_)) != static_cast<int>(kHeight)) {
      throw std::runtime_error("The active X11 mode must be 1280x720 for this test");
    }

    egl_display_ = eglGetDisplay(x_display_);
    if (!egl_display_) throw std::runtime_error("eglGetDisplay(X11) failed: " + eglError());
    EGLint major = 0, minor = 0;
    if (!eglInitialize(egl_display_, &major, &minor)) {
      throw std::runtime_error("eglInitialize failed: " + eglError());
    }
    const char *vendor = eglQueryString(egl_display_, EGL_VENDOR);
    const char *version = eglQueryString(egl_display_, EGL_VERSION);
    egl_vendor_ = vendor ? vendor : "unavailable";
    egl_version_ = version ? version : "unavailable";

    const EGLint config_attributes[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 0, EGL_NONE};
    EGLConfig config = nullptr;
    EGLint config_count = 0;
    if (!eglChooseConfig(egl_display_, config_attributes, &config, 1, &config_count) ||
        config_count == 0) {
      throw std::runtime_error("eglChooseConfig(X11 window) failed: " + eglError());
    }
    EGLint native_visual = 0;
    if (!eglGetConfigAttrib(egl_display_, config, EGL_NATIVE_VISUAL_ID, &native_visual)) {
      throw std::runtime_error("eglGetConfigAttrib(EGL_NATIVE_VISUAL_ID) failed");
    }

    XVisualInfo template_info{};
    template_info.visualid = static_cast<VisualID>(native_visual);
    int visual_count = 0;
    XVisualInfo *visual = XGetVisualInfo(x_display_, VisualIDMask, &template_info, &visual_count);
    if (!visual || visual_count == 0) throw std::runtime_error("No X11 visual matches EGL config");

    XSetWindowAttributes attributes{};
    attributes.colormap = XCreateColormap(x_display_, RootWindow(x_display_, visual->screen),
                                          visual->visual, AllocNone);
    attributes.event_mask = StructureNotifyMask | ExposureMask | KeyPressMask | ButtonPressMask;
    attributes.override_redirect = True;
    x_window_ = XCreateWindow(x_display_, RootWindow(x_display_, visual->screen), 0, 0,
                              kWidth, kHeight, 0, visual->depth, InputOutput, visual->visual,
                              CWColormap | CWEventMask | CWOverrideRedirect, &attributes);
    XFree(visual);
    if (!x_window_) throw std::runtime_error("XCreateWindow failed");
    XStoreName(x_display_, x_window_, "Mali EGL RGB565 benchmark");
    XMapRaised(x_display_, x_window_);
    XSetInputFocus(x_display_, x_window_, RevertToParent, CurrentTime);
    XSync(x_display_, False);

    egl_surface_ = eglCreateWindowSurface(egl_display_, config, x_window_, nullptr);
    if (!egl_surface_) throw std::runtime_error("eglCreateWindowSurface failed: " + eglError());
    if (!eglBindAPI(EGL_OPENGL_ES_API)) throw std::runtime_error("eglBindAPI(OpenGL ES) failed");
    const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    egl_context_ = eglCreateContext(egl_display_, config, nullptr, context_attributes);
    if (!egl_context_) throw std::runtime_error("eglCreateContext failed: " + eglError());
    if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
      throw std::runtime_error("eglMakeCurrent failed: " + eglError());
    }
    if (!eglSwapInterval(egl_display_, 1)) {
      throw std::runtime_error("eglSwapInterval(1) failed: " + eglError());
    }

    gl_renderer_ = glString(GL_RENDERER);
    gl_version_ = glString(GL_VERSION);
    if (gl_renderer_.find("Mali") == std::string::npos) {
      throw std::runtime_error("Refusing to benchmark a non-Mali renderer: " + gl_renderer_);
    }

    const char *vertex_source =
        "attribute vec2 a_position;\n"
        "attribute vec2 a_texcoord;\n"
        "uniform vec4 u_crop;\n"
        "varying vec2 v_texcoord;\n"
        "void main() { gl_Position = vec4(a_position, 0.0, 1.0); "
        "v_texcoord = vec2(mix(u_crop.x, u_crop.z, a_texcoord.x), "
        "mix(u_crop.y, u_crop.w, a_texcoord.y)); }\n";
    const char *fragment_source =
        "precision mediump float;\n"
        "varying vec2 v_texcoord;\n"
        "uniform sampler2D u_frame;\n"
        "void main() { gl_FragColor = texture2D(u_frame, v_texcoord); }\n";
    const GLuint vertex = compileShader(GL_VERTEX_SHADER, vertex_source);
    const GLuint fragment = compileShader(GL_FRAGMENT_SHADER, fragment_source);
    program_ = glCreateProgram();
    glAttachShader(program_, vertex);
    glAttachShader(program_, fragment);
    glBindAttribLocation(program_, 0, "a_position");
    glBindAttribLocation(program_, 1, "a_texcoord");
    glLinkProgram(program_);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    GLint linked = 0;
    glGetProgramiv(program_, GL_LINK_STATUS, &linked);
    if (!linked) {
      GLint length = 0;
      glGetProgramiv(program_, GL_INFO_LOG_LENGTH, &length);
      std::vector<char> log(static_cast<size_t>(std::max(1, length)));
      glGetProgramInfoLog(program_, length, nullptr, log.data());
      throw std::runtime_error(std::string("GL program link failed: ") + log.data());
    }
    position_location_ = glGetAttribLocation(program_, "a_position");
    texcoord_location_ = glGetAttribLocation(program_, "a_texcoord");
    sampler_location_ = glGetUniformLocation(program_, "u_frame");
    crop_location_ = glGetUniformLocation(program_, "u_crop");
    if (position_location_ < 0 || texcoord_location_ < 0 || sampler_location_ < 0 ||
        crop_location_ < 0) {
      throw std::runtime_error("GL shader attributes or uniforms are unavailable");
    }
    const char *box_vertex_source =
        "attribute vec2 a_position;\n"
        "attribute vec4 a_color;\n"
        "varying vec4 v_color;\n"
        "void main() { gl_Position = vec4(a_position, 0.0, 1.0); v_color = a_color; }\n";
    const char *box_fragment_source =
        "precision mediump float;\n"
        "varying vec4 v_color;\n"
        "void main() { gl_FragColor = v_color; }\n";
    const GLuint box_vertex = compileShader(GL_VERTEX_SHADER, box_vertex_source);
    const GLuint box_fragment = compileShader(GL_FRAGMENT_SHADER, box_fragment_source);
    box_program_ = glCreateProgram();
    glAttachShader(box_program_, box_vertex);
    glAttachShader(box_program_, box_fragment);
    glBindAttribLocation(box_program_, 0, "a_position");
    glBindAttribLocation(box_program_, 1, "a_color");
    glLinkProgram(box_program_);
    glDeleteShader(box_vertex);
    glDeleteShader(box_fragment);
    GLint box_linked = 0;
    glGetProgramiv(box_program_, GL_LINK_STATUS, &box_linked);
    if (!box_linked) throw std::runtime_error("Detection overlay shader link failed");
    box_position_location_ = glGetAttribLocation(box_program_, "a_position");
    box_color_location_ = glGetAttribLocation(box_program_, "a_color");
    if (box_position_location_ < 0 || box_color_location_ < 0) {
      throw std::runtime_error("Detection overlay shader inputs are unavailable");
    }

    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, kWidth, kHeight, 0, GL_RGB,
                 GL_UNSIGNED_SHORT_5_6_5, nullptr);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    if (glGetError() != GL_NO_ERROR) throw std::runtime_error("GL texture setup failed");
  }

  void cleanup() {
    if (egl_display_ && egl_context_) eglMakeCurrent(egl_display_, nullptr, nullptr, nullptr);
    if (texture_) glDeleteTextures(1, &texture_);
    if (ui_texture_) glDeleteTextures(1, &ui_texture_);
    if (status_texture_) glDeleteTextures(1, &status_texture_);
    if (dynamic_ui_texture_) glDeleteTextures(1, &dynamic_ui_texture_);
    if (label_texture_) glDeleteTextures(1, &label_texture_);
    if (program_) glDeleteProgram(program_);
    if (box_program_) glDeleteProgram(box_program_);
    if (egl_display_ && egl_surface_) eglDestroySurface(egl_display_, egl_surface_);
    if (egl_display_ && egl_context_) eglDestroyContext(egl_display_, egl_context_);
    if (egl_display_) eglTerminate(egl_display_);
    if (x_display_ && x_window_) XDestroyWindow(x_display_, x_window_);
    if (x_display_) XCloseDisplay(x_display_);
  }

  Display *x_display_ = nullptr;
  Window x_window_ = 0;
  EGLDisplay egl_display_ = nullptr;
  EGLSurface egl_surface_ = nullptr;
  EGLContext egl_context_ = nullptr;
  GLuint program_ = 0;
  GLuint texture_ = 0;
  GLuint ui_texture_ = 0;
  GLuint status_texture_ = 0;
  GLuint dynamic_ui_texture_ = 0;
  GLuint label_texture_ = 0;
  struct HmiSignature {
    dev_t device;
    ino_t inode;
    time_t seconds;
    long nanoseconds;
    uint64_t bytes;
    bool operator==(const HmiSignature &other) const {
      return device == other.device && inode == other.inode && seconds == other.seconds &&
             nanoseconds == other.nanoseconds && bytes == other.bytes;
    }
  };
  HmiSignature dynamic_signature_{};
  bool has_dynamic_signature_ = false;
  uint64_t dynamic_hmi_frame_id_ = 0;
  uint64_t dynamic_hmi_timestamp_ns_ = 0;
  uint64_t status_frame_id_ = UINT64_MAX;
  uint64_t label_frame_id_ = 0;
  bool has_label_frame_ = false;
  GLuint box_program_ = 0;
  GLint position_location_ = -1;
  GLint texcoord_location_ = -1;
  GLint sampler_location_ = -1;
  GLint crop_location_ = -1;
  GLint box_position_location_ = -1;
  GLint box_color_location_ = -1;
  std::array<GLfloat, 4> crop_{0.f, 0.f, 1.f, 1.f};
  std::string egl_vendor_;
  std::string egl_version_;
  std::string gl_renderer_;
  std::string gl_version_;
};

constexpr GLfloat EglX11Window::vertices_[16];

struct Distribution {
  size_t samples = 0;
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double mean = 0.0;
};

Distribution distribution(std::vector<double> values) {
  Distribution result;
  if (values.empty()) return result;
  std::sort(values.begin(), values.end());
  result.samples = values.size();
  const auto quantile = [&values](double q) {
    const double position = q * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] + fraction * (values[upper] - values[lower]);
  };
  result.p50 = quantile(0.50);
  result.p95 = quantile(0.95);
  result.p99 = quantile(0.99);
  double sum = 0.0;
  for (double value : values) sum += value;
  result.mean = sum / static_cast<double>(values.size());
  return result;
}

uint64_t readNumber(const std::string &path) {
  std::ifstream input(path);
  uint64_t value = 0;
  input >> value;
  return input ? value : 0;
}

int readGpuLoad() {
  std::ifstream input("/sys/class/devfreq/fde60000.gpu/load");
  std::string value;
  if (!input || !(input >> value)) return -1;
  try {
    return std::stoi(value.substr(0, value.find('@')));
  } catch (...) {
    return -1;
  }
}

struct ThermalZone {
  std::string name;
  std::string path;
  std::vector<double> celsius;
};

class TelemetrySampler {
 public:
  TelemetrySampler() {
    for (int i = 0; i < 16; ++i) {
      const std::string base = "/sys/class/thermal/thermal_zone" + std::to_string(i);
      std::ifstream type(base + "/type");
      std::string name;
      if (type >> name) zones_.push_back(ThermalZone{name, base + "/temp", {}});
    }
  }

  void start(uint32_t duration_seconds) {
    const size_t capacity = static_cast<size_t>(duration_seconds) * 4U + 2U;
    gpu_frequency_mhz_.reserve(capacity);
    gpu_load_percent_.reserve(capacity);
    for (auto &zone : zones_) zone.celsius.reserve(capacity);
    running_.store(true);
    worker_ = std::thread([this]() {
      while (running_.load()) {
        const uint64_t frequency = readNumber("/sys/class/devfreq/fde60000.gpu/cur_freq");
        const int load = readGpuLoad();
        if (frequency) gpu_frequency_mhz_.push_back(static_cast<double>(frequency) / 1000000.0);
        if (load >= 0) gpu_load_percent_.push_back(load);
        for (auto &zone : zones_) {
          const uint64_t milli_celsius = readNumber(zone.path);
          if (milli_celsius) zone.celsius.push_back(static_cast<double>(milli_celsius) / 1000.0);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
      }
    });
  }

  void stop() {
    running_.store(false);
    if (worker_.joinable()) worker_.join();
  }

  ~TelemetrySampler() { stop(); }
  const std::vector<double> &gpuFrequencyMhz() const { return gpu_frequency_mhz_; }
  const std::vector<double> &gpuLoadPercent() const { return gpu_load_percent_; }
  const std::vector<ThermalZone> &zones() const { return zones_; }

 private:
  std::atomic<bool> running_{false};
  std::thread worker_;
  std::vector<double> gpu_frequency_mhz_;
  std::vector<double> gpu_load_percent_;
  std::vector<ThermalZone> zones_;
};

class AiBridgeClient {
 public:
  AiBridgeClient(std::string frame_file, uint16_t request_port, uint16_t result_port)
      : frame_file_(std::move(frame_file)) {
    request_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (request_fd_ < 0) throw std::runtime_error("Cannot create AI request socket");
    result_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (result_fd_ < 0) {
      close(request_fd_);
      request_fd_ = -1;
      throw std::runtime_error("Cannot create AI result socket");
    }
    const int reuse = 1;
    setsockopt(result_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    const int current_flags = fcntl(result_fd_, F_GETFL, 0);
    if (current_flags >= 0) fcntl(result_fd_, F_SETFL, current_flags | O_NONBLOCK);

    request_address_.sin_family = AF_INET;
    request_address_.sin_port = htons(request_port);
    request_address_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    sockaddr_in result_address{};
    result_address.sin_family = AF_INET;
    result_address.sin_port = htons(result_port);
    result_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(result_fd_, reinterpret_cast<const sockaddr *>(&result_address),
             sizeof(result_address)) != 0) {
      close(request_fd_);
      close(result_fd_);
      request_fd_ = result_fd_ = -1;
      throw std::runtime_error("Cannot bind AI result UDP port");
    }
  }

  ~AiBridgeClient() {
    if (request_fd_ >= 0) close(request_fd_);
    if (result_fd_ >= 0) close(result_fd_);
  }

  bool request(const FrameView &frame, double input_fps, double submitted_fps) {
    if (!frame.pixels || frame.stride_bytes != kStrideBytes) {
      ++requests_dropped_;
      return false;
    }
    const float input_rate = static_cast<float>(input_fps);
    const float submitted_rate = static_cast<float>(submitted_fps);
    if (!std::isfinite(input_rate) || !std::isfinite(submitted_rate) ||
        input_rate < 0.f || submitted_rate < 0.f) {
      ++requests_dropped_;
      return false;
    }
    const std::string temporary_path = frame_file_ + ".tmp";
    std::ofstream snapshot(temporary_path, std::ios::binary | std::ios::trunc);
    if (!snapshot) {
      ++requests_dropped_;
      return false;
    }
    std::array<unsigned char, 28> snapshot_header{};
    std::memcpy(snapshot_header.data(), "MFR1", 4);
    std::memcpy(snapshot_header.data() + 4, &frame.frame_id, sizeof(frame.frame_id));
    std::memcpy(snapshot_header.data() + 12, &frame.timestamp_ns, sizeof(frame.timestamp_ns));
    std::memcpy(snapshot_header.data() + 20, &frame.stride_bytes, sizeof(frame.stride_bytes));
    constexpr uint32_t payload_bytes = static_cast<uint32_t>(kFrameBytes);
    std::memcpy(snapshot_header.data() + 24, &payload_bytes, sizeof(payload_bytes));
    snapshot.write(reinterpret_cast<const char *>(snapshot_header.data()), snapshot_header.size());
    snapshot.write(reinterpret_cast<const char *>(frame.pixels), static_cast<std::streamsize>(kFrameBytes));
    snapshot.close();
    if (!snapshot || rename(temporary_path.c_str(), frame_file_.c_str()) != 0) {
      unlink(temporary_path.c_str());
      ++requests_dropped_;
      return false;
    }

    std::array<unsigned char, 28> packet{};
    std::memcpy(packet.data(), "MVR2", 4);
    std::memcpy(packet.data() + 4, &frame.frame_id, sizeof(frame.frame_id));
    std::memcpy(packet.data() + 12, &frame.timestamp_ns, sizeof(frame.timestamp_ns));
    std::memcpy(packet.data() + 20, &input_rate, sizeof(input_rate));
    std::memcpy(packet.data() + 24, &submitted_rate, sizeof(submitted_rate));
    const ssize_t sent = sendto(
        request_fd_, packet.data(), packet.size(), MSG_DONTWAIT,
        reinterpret_cast<const sockaddr *>(&request_address_), sizeof(request_address_));
    if (sent == static_cast<ssize_t>(packet.size())) {
      ++requests_sent_;
      return true;
    }
    ++requests_dropped_;
    return false;
  }

  bool receiveLatest() {
    constexpr size_t kV1HeaderBytes = 36;
    constexpr size_t kV2HeaderBytes = 40;
    constexpr size_t kV1BoxBytes = 28;
    constexpr size_t kV2BoxBytes = 36;
    constexpr size_t kMaxStatusBytes = 4096;
    std::array<unsigned char, kV2HeaderBytes + kMaxDetectionBoxes * kV2BoxBytes +
                                  kMaxStatusBytes> bytes{};
    bool received_any = false;
    while (true) {
      const ssize_t received = recv(result_fd_, bytes.data(), bytes.size(), MSG_DONTWAIT);
      if (received < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) break;
        break;
      }
      const bool version_two = std::memcmp(bytes.data(), "MVA2", 4) == 0;
      if (received < static_cast<ssize_t>(version_two ? kV2HeaderBytes : kV1HeaderBytes) ||
          (!version_two && std::memcmp(bytes.data(), "MVA1", 4) != 0)) {
        continue;
      }
      const size_t header_bytes = version_two ? kV2HeaderBytes : kV1HeaderBytes;
      const size_t box_bytes = version_two ? kV2BoxBytes : kV1BoxBytes;
      uint32_t count = 0;
      uint32_t status_bytes = 0;
      DetectionPacket packet;
      std::memcpy(&packet.frame_id, bytes.data() + 4, sizeof(packet.frame_id));
      std::memcpy(&packet.timestamp_ns, bytes.data() + 12, sizeof(packet.timestamp_ns));
      std::memcpy(&count, bytes.data() + 20, sizeof(count));
      std::memcpy(&packet.flags, bytes.data() + 24, sizeof(packet.flags));
      std::memcpy(&packet.nav_direction, bytes.data() + 28, sizeof(packet.nav_direction));
      std::memcpy(&packet.nav_distance_m, bytes.data() + 32, sizeof(packet.nav_distance_m));
      if (version_two) std::memcpy(&status_bytes, bytes.data() + 36, sizeof(status_bytes));
      if (count > kMaxDetectionBoxes ||
          status_bytes > kMaxStatusBytes ||
          received != static_cast<ssize_t>(header_bytes + count * box_bytes + status_bytes) ||
          (has_latest_ && packet.frame_id < latest_.frame_id)) {
        continue;
      }
      packet.count = count;
      bool valid = true;
      for (size_t index = 0; index < packet.count; ++index) {
        float values[7]{};
        std::memcpy(values, bytes.data() + header_bytes + index * box_bytes,
                    sizeof(values));
        for (float value : values) valid = valid && std::isfinite(value);
        uint32_t class_id = 255;
        float confidence = 0.f;
        if (version_two) {
          const size_t box_offset = header_bytes + index * box_bytes;
          std::memcpy(&class_id, bytes.data() + box_offset + 28, sizeof(class_id));
          std::memcpy(&confidence, bytes.data() + box_offset + 32, sizeof(confidence));
          valid = valid && std::isfinite(confidence) && confidence >= 0.f && confidence <= 1.f;
        }
        packet.boxes[index] = DetectionBox{values[0], values[1], values[2], values[3],
                                           values[4], values[5], values[6], class_id, confidence};
      }
      if (!valid || !std::isfinite(packet.nav_distance_m)) continue;
      latest_ = packet;
      has_latest_ = true;
      ++results_received_;
      received_any = true;
    }
    return received_any;
  }

  bool snapshot(uint64_t current_frame_id, uint64_t now_ns, DetectionPacket &packet) const {
    if (!has_latest_ || current_frame_id < latest_.frame_id ||
        current_frame_id - latest_.frame_id > 15 || now_ns < latest_.timestamp_ns ||
        now_ns - latest_.timestamp_ns > 250000000ULL) {
      return false;
    }
    packet = latest_;
    return true;
  }

  uint64_t requestsSent() const { return requests_sent_; }
  uint64_t requestsDropped() const { return requests_dropped_; }
  uint64_t resultsReceived() const { return results_received_; }
  const DetectionPacket &latest() const { return latest_; }

 private:
  std::string frame_file_;
  int request_fd_ = -1;
  int result_fd_ = -1;
  sockaddr_in request_address_{};
  DetectionPacket latest_{};
  bool has_latest_ = false;
  uint64_t requests_sent_ = 0;
  uint64_t requests_dropped_ = 0;
  uint64_t results_received_ = 0;
};

uint64_t currentRssKb() {
  std::ifstream input("/proc/self/status");
  std::string key;
  while (input >> key) {
    if (key == "VmRSS:") {
      uint64_t value = 0;
      input >> value;
      return value;
    }
    std::string rest;
    std::getline(input, rest);
  }
  return 0;
}

std::string jsonEscape(const std::string &value) {
  std::string out;
  for (char c : value) {
    if (c == '"' || c == '\\') out.push_back('\\');
    if (static_cast<unsigned char>(c) >= 0x20) out.push_back(c);
  }
  return out;
}

void writeDistribution(std::ostream &out, const Distribution &value, const char *unit = "ms") {
  out << "{\"samples\":" << value.samples << ",\"p50_" << unit << "\":" << value.p50
      << ",\"p95_" << unit << "\":" << value.p95 << ",\"p99_" << unit << "\":"
      << value.p99 << ",\"mean_" << unit << "\":"
      << value.mean << '}';
}

double frequencyFromTimestamps(const std::vector<uint64_t> &timestamps_ns) {
  if (timestamps_ns.size() < 2 || timestamps_ns.back() <= timestamps_ns.front()) return 0.0;
  return static_cast<double>(timestamps_ns.size() - 1) * 1e9 /
         static_cast<double>(timestamps_ns.back() - timestamps_ns.front());
}

struct Timings {
  std::vector<double> input_interval_ms;
  std::vector<double> submitted_interval_ms;
  std::vector<double> swap_interval_ms;
  std::vector<uint64_t> submitted_times_ns;
  std::vector<uint64_t> swap_times_ns;
  std::vector<double> source_wait_ms;
  std::vector<double> source_age_ms;
  std::vector<double> texture_upload_ms;
  std::vector<double> draw_submit_ms;
  std::vector<double> static_ui_draw_ms;
  std::vector<double> dynamic_hmi_draw_ms;
  std::vector<double> dynamic_hmi_upload_ms;
  std::vector<double> detection_layer_draw_ms;
  std::vector<double> detection_age_ms;
  std::vector<double> ai_link_ms;
  std::vector<double> egl_swap_ms;
  std::vector<double> frame_cpu_ms;
};

struct Options {
  std::string source = "synthetic";
  std::vector<std::string> files;
  std::string manifest;
  std::string output_json;
  std::string pgv_file = "/dev/shm/pgv_live_codex_20261018_postreboot/latest.pgv";
  std::string ai_frame_file = "/dev/shm/mali_video_ai_frame.rgb565";
  std::string dynamic_hmi_file = "/dev/shm/mali_video_dynamic_hmi.rgba";
  std::string design_preview_file;
  std::string layout = "quad";
  std::string view = "quad";
  std::string mode = "MULTI";
  uint32_t road_quadrant = 3;
  uint32_t navigation_quadrant = 1;
  uint32_t fps = 60;
  uint32_t duration_seconds = 60;
  uint32_t ai_request_hz = 10;
  bool overlays = false;
  bool demo_boxes = false;
  bool ai_bridge = false;
  bool layout_explicit = false;
  bool view_explicit = false;
  bool mode_explicit = false;
  bool design_preview = false;
};

Options parseOptions(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char *name) -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(std::string("Missing value for ") + name);
      return argv[++i];
    };
    if (arg == "--source") options.source = value("--source");
    else if (arg == "--file") options.files.push_back(value("--file"));
    else if (arg == "--manifest") options.manifest = value("--manifest");
    else if (arg == "--pgv-file") options.pgv_file = value("--pgv-file");
    else if (arg == "--ai-frame-file") options.ai_frame_file = value("--ai-frame-file");
    else if (arg == "--dynamic-hmi-file") options.dynamic_hmi_file = value("--dynamic-hmi-file");
    else if (arg == "--design-preview-file") {
      options.design_preview_file = value("--design-preview-file");
      options.design_preview = true;
      options.overlays = true;
    }
    else if (arg == "--layout") {
      options.layout = value("--layout");
      options.layout_explicit = true;
    } else if (arg == "--view") {
      options.view = value("--view");
      options.view_explicit = true;
    } else if (arg == "--mode") {
      options.mode = value("--mode");
      options.mode_explicit = true;
    } else if (arg == "--road-quadrant") {
      options.road_quadrant = static_cast<uint32_t>(std::stoul(value("--road-quadrant")));
    } else if (arg == "--navigation-quadrant") {
      options.navigation_quadrant =
          static_cast<uint32_t>(std::stoul(value("--navigation-quadrant")));
    } else if (arg == "--ai-request-hz") {
      options.ai_request_hz = static_cast<uint32_t>(std::stoul(value("--ai-request-hz")));
    }
    else if (arg == "--fps") options.fps = static_cast<uint32_t>(std::stoul(value("--fps")));
    else if (arg == "--duration") {
      options.duration_seconds = static_cast<uint32_t>(std::stoul(value("--duration")));
    } else if (arg == "--output-json") options.output_json = value("--output-json");
    else if (arg == "--overlays") options.overlays = true;
    else if (arg == "--demo-boxes") {
      options.demo_boxes = true;
      options.overlays = true;
    }
    else if (arg == "--ai-bridge") {
      options.ai_bridge = true;
      options.overlays = true;
    }
    else if (arg == "--help") {
      std::cout << "Usage: mali_video [--source synthetic|replay|pgv-live] "
                   "[--file FRAME.rgb565 ...|--manifest replay.txt] [--pgv-file latest.pgv] [--layout quad|front] "
                   "[--view quad|front|spatial] [--mode FOCUS|MULTI|SPATIAL] "
                   "[--road-quadrant 0..3] [--navigation-quadrant 0..3] [--fps 60] [--duration 60] "
                   "[--overlays] [--demo-boxes|--ai-bridge] [--ai-frame-file FILE] "
                   "[--dynamic-hmi-file FILE] [--design-preview-file FILE] [--output-json FILE]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("Unknown option: " + arg);
    }
  }
  if (options.source != "synthetic" && options.source != "replay" &&
      options.source != "pgv-live") {
    throw std::invalid_argument("Source must be synthetic, replay, or pgv-live");
  }
  if (options.fps < 1 || options.fps > 240 || options.duration_seconds < 1) {
    throw std::invalid_argument("FPS must be 1..240 and duration must be positive");
  }
  if (options.ai_request_hz < 1 || options.ai_request_hz > options.fps) {
    throw std::invalid_argument("AI request rate must be between 1 and the display rate");
  }
  if (options.layout != "quad" && options.layout != "front") {
    throw std::invalid_argument("Layout must be quad or front");
  }
  if (options.view != "quad" && options.view != "front" && options.view != "spatial") {
    throw std::invalid_argument("View must be quad, front, or spatial");
  }
  if (options.road_quadrant > 3) throw std::invalid_argument("Road quadrant must be 0..3");
  if (options.navigation_quadrant > 3) {
    throw std::invalid_argument("Navigation quadrant must be 0..3");
  }
  if (options.mode != "FOCUS" && options.mode != "MULTI" && options.mode != "SPATIAL") {
    throw std::invalid_argument("Mode must be FOCUS, MULTI, or SPATIAL");
  }
  if (options.source == "synthetic") {
    if (!options.layout_explicit) options.layout = "front";
  }
  if (options.mode_explicit) {
    options.view = options.mode == "FOCUS" ? "front" : options.mode == "SPATIAL" ? "spatial" : "quad";
  } else if (options.view_explicit) {
    options.mode = options.view == "front" ? "FOCUS" : options.view == "spatial" ? "SPATIAL" : "MULTI";
  } else if (options.source == "synthetic") {
    options.view = "front";
    options.mode = "FOCUS";
  } else {
    options.view = "quad";
    options.mode = "MULTI";
  }
  if (options.source == "replay" && options.files.empty() && options.manifest.empty()) {
    throw std::invalid_argument("Replay source needs --file or --manifest");
  }
  if (!options.files.empty() && !options.manifest.empty()) {
    throw std::invalid_argument("Replay source accepts either --file or --manifest, not both");
  }
  if (options.layout == "front" && options.view == "quad") {
    throw std::invalid_argument("A front-only source cannot be displayed as a four-camera view");
  }
  if (options.ai_bridge && options.demo_boxes) {
    throw std::invalid_argument("Choose either demo boxes or the AI bridge");
  }
  if (options.design_preview && options.ai_bridge) {
    throw std::invalid_argument("DESIGN_PREVIEW is isolated from live AI; omit --ai-bridge");
  }
  return options;
}

std::unique_ptr<FrameSource> createSource(const Options &options) {
  if (options.source == "replay") {
    if (!options.manifest.empty()) return std::make_unique<ReplayFrameSource>(options.manifest);
    return std::make_unique<ReplayFrameSource>(options.files);
  }
  if (options.source == "pgv-live") return std::make_unique<PgvLiveFrameSource>(options.pgv_file);
  return std::make_unique<SyntheticFrameSource>();
}

void writeSummary(const std::string &path, const Options &options, const FrameSource &source,
                  const EglX11Window &window, const Timings &timings, uint64_t duration_ns,
                  uint64_t cpu_ns, uint64_t rss_peak_kb, uint64_t rss_end_kb,
                   uint64_t received, uint64_t submitted, uint64_t swaps,
                   uint64_t ui_only_swaps,
                  uint64_t dropped_ids,
                  const TelemetrySampler &telemetry,
                  const SyntheticDetectionWorker *demo_worker, uint64_t box_frames_drawn,
                  const AiBridgeClient *ai_bridge, double static_ui_prepare_ms,
                  const PgvLiveFrameSource *pgv_source) {
  std::ostringstream report;
  report << std::fixed << std::setprecision(3);
  const double duration_s = static_cast<double>(duration_ns) / 1e9;
  const uint64_t produced = source.publishedCount();
  const double input_fps = frequencyFromTimestamps(source.publishTimesNs());
  const double submitted_fps = frequencyFromTimestamps(timings.submitted_times_ns);
  const double swap_fps = frequencyFromTimestamps(timings.swap_times_ns);
  const long cpus = std::max<long>(1, sysconf(_SC_NPROCESSORS_ONLN));
  report << "{\n"
         << "  \"evidence\": \"" << source.name() << "\",\n"
         << "  \"source\": \"" << source.name() << "\",\n"
         << "  \"duration_s\": " << duration_s << ",\n"
         << "  \"target_input_fps\": " << options.fps << ",\n"
         << "  \"source_frames_produced\": " << produced << ",\n"
         << "  \"source_frames_received\": " << received << ",\n"
         << "  \"actual_input_fps\": " << input_fps << ",\n"
         << "  \"unique_source_frames_submitted\": " << submitted << ",\n"
         << "  \"unique_source_submit_fps\": " << submitted_fps << ",\n"
         << "  \"egl_swap_interval\": 1,\n"
         << "  \"egl_swap_calls\": " << swaps << ",\n"
         << "  \"egl_swap_calls_per_s\": " << swap_fps << ",\n"
         << "  \"ui_only_swap_calls\": " << ui_only_swaps << ",\n"
         << "  \"source_ids_skipped_by_latest_only_consumer\": " << dropped_ids << ",\n"
         << "  \"actual_scanout_fps\": null,\n"
         << "  \"scanout_measurement\": \"Not measured; EGL swap completion is not proof of distinct HDMI scanout frames.\",\n"
         << "  \"video\": {\"width\": " << kWidth << ", \"height\": " << kHeight
         << ", \"pixel_format\": \"RGB565_LE\", \"row_stride_bytes\": " << kStrideBytes
         << ", \"frame_bytes\": " << kFrameBytes << ", \"dma_slot_bytes\": "
         << kPgvBufferBytes << "},\n"
         << "  \"renderer\": {\"egl_vendor\": \"" << jsonEscape(window.eglVendor())
         << "\", \"egl_version\": \"" << jsonEscape(window.eglVersion())
         << "\", \"gles_renderer\": \"" << jsonEscape(window.glRenderer())
         << "\", \"gles_version\": \"" << jsonEscape(window.glVersion()) << "\"},\n"
         << "  \"source_prepare_ms\": " << source.prepareMs() << ",\n"
         << "  \"synthetic_rgb565_color_readback\": \""
         << (source.name() == "SYNTHETIC" ? "PASS" : "not_run") << "\",\n"
         << "  \"static_hmi_ui\": \"" << (options.overlays ? "CACHED_RGBA_TEXTURE" : "disabled")
         << "\",\n"
         << "  \"static_hmi_cache_prepare_ms\": " << static_ui_prepare_ms << ",\n"
         << "  \"source_layout\": \"" << options.layout << "\",\n"
         << "  \"display_view\": \"" << options.view << "\",\n"
         << "  \"display_mode\": \"" << options.mode << "\",\n"
         << "  \"focus_quadrant\": " << options.road_quadrant << ",\n"
         << "  \"navigation_quadrant\": " << options.navigation_quadrant << ",\n"
         << "  \"replay_timestamp_mode\": \""
         << (dynamic_cast<const ReplayFrameSource *>(&source)
                 ? dynamic_cast<const ReplayFrameSource *>(&source)->timestampMode()
                 : "not_applicable")
         << "\",\n"
         << "  \"replay_manifest\": \""
         << (dynamic_cast<const ReplayFrameSource *>(&source)
                 ? jsonEscape(dynamic_cast<const ReplayFrameSource *>(&source)->manifestPath())
                 : "")
         << "\",\n"
         << "  \"road_quadrant\": " << options.road_quadrant << ",\n"
         << "  \"pgv_file\": \"" << (pgv_source ? jsonEscape(pgv_source->path()) : "not_used")
         << "\",\n"
         << "  \"pgv_source_frames_skipped\": "
         << (pgv_source ? pgv_source->sourceFramesSkipped() : 0) << ",\n"
         << "  \"pgv_duplicate_frames\": " << (pgv_source ? pgv_source->duplicateFrames() : 0)
         << ",\n"
         << "  \"pgv_invalid_frames\": " << (pgv_source ? pgv_source->invalidFrames() : 0)
         << ",\n"
         << "  \"pgv_buffer_busy_drops\": " << (pgv_source ? pgv_source->bufferBusyDrops() : 0)
         << ",\n"
         << "  \"pgv_unavailable_polls\": " << (pgv_source ? pgv_source->unavailableSamples() : 0)
         << ",\n"
         << "  \"detection_overlay_provider\": \""
          << (ai_bridge ? "RKNN_ASYNC_EXACT_SOURCE_SNAPSHOT"
                        : options.demo_boxes ? "SYNTHETIC_DEMO_ASYNC" : "disabled")
          << "\",\n"
          << "  \"ai_inference_enabled\": "
          << (ai_bridge && ai_bridge->resultsReceived() > 0 ? "true" : "false") << ",\n"
          << "  \"ai_request_frames_sent\": "
          << (ai_bridge ? ai_bridge->requestsSent() : 0) << ",\n"
          << "  \"ai_request_frames_dropped\": "
          << (ai_bridge ? ai_bridge->requestsDropped() : 0) << ",\n"
          << "  \"ai_result_packets_received\": "
          << (ai_bridge ? ai_bridge->resultsReceived() : 0) << ",\n"
          << "  \"ai_latest_result_flags\": "
          << (ai_bridge ? ai_bridge->latest().flags : 0) << ",\n"
          << "  \"ai_latest_navigation_direction_code\": "
          << (ai_bridge ? static_cast<unsigned>(ai_bridge->latest().nav_direction) : 0) << ",\n"
          << "  \"ai_latest_navigation_distance_m\": "
          << (ai_bridge ? ai_bridge->latest().nav_distance_m : -1.f) << ",\n"
         << "  \"demo_detection_frames_analyzed\": "
         << (demo_worker ? demo_worker->analyzedFrames() : 0) << ",\n"
         << "  \"demo_detection_source_ids_skipped\": "
         << (demo_worker ? demo_worker->skippedSourceIds() : 0) << ",\n"
         << "  \"frames_with_detection_overlay\": " << box_frames_drawn << ",\n"
         << "  \"source_input_interval_ms\": ";
  writeDistribution(report, distribution(timings.input_interval_ms));
  report << ",\n  \"unique_source_submit_interval_ms\": ";
  writeDistribution(report, distribution(timings.submitted_interval_ms));
  report << ",\n  \"egl_swap_interval_ms\": ";
  writeDistribution(report, distribution(timings.swap_interval_ms));
  report << ",\n  \"stage_latency_ms\": {\n"
         << "    \"source_acquire_or_generation\": ";
  writeDistribution(report, distribution(source.acquireTimesMs()));
  report << ",\n"
         << "    \"source_wait_for_next_input\": ";
  writeDistribution(report, distribution(timings.source_wait_ms));
  report << ",\n    \"source_to_submit_age\": ";
  writeDistribution(report, distribution(timings.source_age_ms));
  report << ",\n    \"rgb565_texture_upload_call\": ";
  writeDistribution(report, distribution(timings.texture_upload_ms));
  report << ",\n    \"draw_call_submission\": ";
  writeDistribution(report, distribution(timings.draw_submit_ms));
  report << ",\n    \"cached_static_hmi_draw_call\": ";
  writeDistribution(report, distribution(timings.static_ui_draw_ms));
  report << ",\n    \"dynamic_hmi_texture_upload_call\": ";
  writeDistribution(report, distribution(timings.dynamic_hmi_upload_ms));
  report << ",\n    \"dynamic_hmi_draw_call\": ";
  writeDistribution(report, distribution(timings.dynamic_hmi_draw_ms));
  report << ",\n    \"detection_box_draw_call\": ";
  writeDistribution(report, distribution(timings.detection_layer_draw_ms));
  report << ",\n    \"detection_metadata_age\": ";
  writeDistribution(report, distribution(timings.detection_age_ms));
  report << ",\n    \"ai_request_send_and_result_poll\": ";
  writeDistribution(report, distribution(timings.ai_link_ms));
  report << ",\n    \"egl_swap_blocking_call\": ";
  writeDistribution(report, distribution(timings.egl_swap_ms));
  report << ",\n    \"upload_draw_swap_total\": ";
  writeDistribution(report, distribution(timings.frame_cpu_ms));
  report << "\n  },\n  \"demo_detection_provider_latency_ms\": ";
  writeDistribution(report, demo_worker ? distribution(demo_worker->latenciesMs())
                                        : Distribution{});
  report << ",\n  \"resources\": {\n"
         << "    \"process_cpu_one_core_percent\": "
         << (duration_ns ? 100.0 * cpu_ns / duration_ns : 0.0) << ",\n"
         << "    \"process_cpu_machine_percent\": "
         << (duration_ns ? 100.0 * cpu_ns / duration_ns / cpus : 0.0) << ",\n"
         << "    \"online_cpus\": " << cpus << ",\n"
         << "    \"process_rss_end_mb\": " << static_cast<double>(rss_end_kb) / 1024.0 << ",\n"
         << "    \"process_rss_peak_mb\": " << static_cast<double>(rss_peak_kb) / 1024.0
         << "\n  },\n  \"gpu_samples\": {\n"
         << "    \"driver_load_percent\": ";
  writeDistribution(report, distribution(telemetry.gpuLoadPercent()), "percent");
  report << ",\n    \"frequency_mhz\": ";
  writeDistribution(report, distribution(telemetry.gpuFrequencyMhz()), "mhz");
  report << ",\n    \"frequency_source\": \"/sys/class/devfreq/fde60000.gpu/cur_freq\",\n"
         << "    \"load_source\": \"/sys/class/devfreq/fde60000.gpu/load\",\n"
         << "    \"timing_note\": \"Stage timings are CPU API-call durations; GPU execution time is represented by kernel load/frequency samples.\"\n"
         << "  },\n  \"temperatures_c\": [";
  bool first_zone = true;
  for (const auto &zone : telemetry.zones()) {
    if (zone.celsius.empty()) continue;
    if (!first_zone) report << ',';
    first_zone = false;
    const auto limits = std::minmax_element(zone.celsius.begin(), zone.celsius.end());
    report << "\n    {\"zone\":\"" << jsonEscape(zone.name) << "\",\"min_c\":"
           << *limits.first << ",\"max_c\":" << *limits.second
           << ",\"end_c\":" << zone.celsius.back() << '}';
  }
  if (!first_zone) report << '\n' << "  ";
  report << "],\n  \"limits\": ["
         << "\"Input evidence is " << source.name() << "; it is not a live PCIe run.\","
         << "\"egl_swap_calls are not measured scanout frames.\","
         << "\"The RGB565 texture is updated by direct GL_UNSIGNED_SHORT_5_6_5 upload without CPU pixel conversion.\""
         << "]\n}\n";
  const std::string json = report.str();
  std::cout << json;
  if (!path.empty()) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot create summary JSON: " + path);
    output << json;
    if (!output) throw std::runtime_error("Cannot write summary JSON: " + path);
  }
}

int run(const Options &options) {
  std::unique_ptr<FrameSource> source = createSource(options);
  std::unique_ptr<AiBridgeClient> ai_bridge;
  if (options.ai_bridge) {
    ai_bridge = std::make_unique<AiBridgeClient>(options.ai_frame_file, 38260, 38261);
  }
  std::unique_ptr<StaticHmiLayer> hmi_layer;
  if (options.overlays && !options.design_preview) {
    hmi_layer = std::make_unique<StaticHmiLayer>(source->name(), options.mode,
                                                 options.road_quadrant,
                                                 options.navigation_quadrant);
  }
  EglX11Window window;
  enum class UiView { Quad, Focus, Spatial };
  UiView current_view = options.mode == "SPATIAL"
                            ? UiView::Spatial
                            : options.mode == "FOCUS" ? UiView::Focus : UiView::Quad;
  uint32_t focus_quadrant = options.road_quadrant;
  const auto apply_view = [&]() {
    window.setVideoCrop(focus_quadrant,
                        options.layout == "quad" && current_view == UiView::Focus);
  };
  apply_view();
  if (hmi_layer) window.setStaticUiPixels(hmi_layer->pixels().data());
  const bool design_ui_loaded = options.design_preview &&
                                window.loadStaticUiFile(options.design_preview_file);
  if (options.design_preview && !design_ui_loaded) {
    throw std::runtime_error("DESIGN_PREVIEW texture could not be loaded: " +
                             options.design_preview_file);
  }
  const bool warmup_dynamic_loaded = options.ai_bridge &&
                                      window.loadDynamicUiFile(options.dynamic_hmi_file);
  FrameView warmup{source->pixelsFor(0), 0, 0, kStrideBytes, {}};
  for (int i = 0; i < 30; ++i) {
    window.upload(warmup.pixels);
    window.draw();
    if (i == 0 && source->name() == "SYNTHETIC") window.verifySyntheticRedPixel();
    if (design_ui_loaded) {
      window.drawVideoThumbnails(options.layout == "quad");
      window.drawStaticUi();
    } else {
      if (hmi_layer) window.drawStaticUi();
      if (warmup_dynamic_loaded) window.drawDynamicUi();
      if (hmi_layer) window.drawVideoThumbnails(options.layout == "quad");
      if (hmi_layer) window.drawSpatialInset(nullptr);
    }
    window.swap();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(250));

  std::cout << "RUNNING evidence=" << source->name() << " duration_s="
            << options.duration_seconds << " target_fps=" << options.fps
            << " renderer=" << window.glRenderer() << std::endl;

  Timings timings;
  const size_t expected_frames =
      static_cast<size_t>(options.duration_seconds) * options.fps + 1U;
  timings.input_interval_ms.reserve(expected_frames);
  timings.submitted_interval_ms.reserve(expected_frames);
  timings.swap_interval_ms.reserve(expected_frames);
  timings.submitted_times_ns.reserve(expected_frames);
  timings.swap_times_ns.reserve(expected_frames + 16U);
  timings.source_wait_ms.reserve(expected_frames);
  timings.source_age_ms.reserve(expected_frames);
  timings.texture_upload_ms.reserve(expected_frames);
  timings.draw_submit_ms.reserve(expected_frames);
  timings.static_ui_draw_ms.reserve(expected_frames);
  timings.dynamic_hmi_draw_ms.reserve(expected_frames);
  timings.dynamic_hmi_upload_ms.reserve(expected_frames);
  timings.detection_layer_draw_ms.reserve(expected_frames);
  timings.detection_age_ms.reserve(expected_frames);
  timings.ai_link_ms.reserve(expected_frames);
  timings.egl_swap_ms.reserve(expected_frames);
  timings.frame_cpu_ms.reserve(expected_frames);
  TelemetrySampler telemetry;
  const uint64_t cpu_start = processCpuNs();
  telemetry.start(options.duration_seconds);
  source->start(options.fps, options.duration_seconds);
  const uint64_t start_ns = source->startNs();
  const uint64_t end_ns =
      start_ns + static_cast<uint64_t>(options.duration_seconds) * 1000000000ULL;
  SyntheticDetectionWorker demo_worker;
  if (options.demo_boxes) {
    demo_worker.start(*source, end_ns, options.fps, options.duration_seconds);
  }
  uint64_t last_frame_id = UINT64_MAX;
  uint64_t submitted = 0;
  uint64_t received = 0;
  uint64_t swaps = 0;
  uint64_t dropped_ids = 0;
  uint64_t box_frames_drawn = 0;
  uint64_t ui_only_swaps = 0;
  uint64_t last_ai_request_ns = 0;
  uint64_t last_idle_redraw_ns = 0;
  bool quit_requested = false;
  const uint64_t ai_period_ns = 1000000000ULL / options.ai_request_hz;
  const uint64_t source_period_ns = 1000000000ULL / options.fps;

  while (!quit_requested && monotonicNs() < end_ns) {
    for (const char command : window.pollControls()) {
      if (command == 'x') {
        quit_requested = true;
      } else if (command == 'q' || command == 'm') {
        current_view = UiView::Quad;
        apply_view();
      } else if (command == 'f') {
        current_view = UiView::Focus;
        apply_view();
      } else if (command == 's') {
        current_view = UiView::Spatial;
        apply_view();
      } else if (command >= '1' && command <= '4') {
        focus_quadrant = static_cast<uint32_t>(command - '1');
        current_view = UiView::Focus;
        apply_view();
      } else if (command == 'n') {
        focus_quadrant = (focus_quadrant + 1U) % 4U;
        current_view = UiView::Focus;
        apply_view();
      }
    }
    if (quit_requested) break;
    const uint64_t acquire_start = monotonicNs();
    FrameView frame;
    const uint64_t poll_deadline = std::min(end_ns, acquire_start + source_period_ns);
    if (!source->waitLatest(last_frame_id, poll_deadline, frame)) {
      if (monotonicNs() < end_ns) {
        const uint64_t idle_now_ns = monotonicNs();
        const bool dynamic_hmi_changed = options.ai_bridge &&
                                         window.loadDynamicUiFile(options.dynamic_hmi_file);
        if (dynamic_hmi_changed || idle_now_ns - last_idle_redraw_ns >= 100000000ULL) {
          window.draw();
          if (design_ui_loaded) {
            window.drawVideoThumbnails(options.layout == "quad");
            window.drawStaticUi();
          } else {
            if (hmi_layer) window.drawStaticUi();
            if (options.ai_bridge) window.drawDynamicUi();
            if (hmi_layer) window.drawVideoThumbnails(options.layout == "quad");
            if (hmi_layer) window.drawSpatialInset(nullptr);
          }
          window.swap();
          timings.swap_times_ns.push_back(monotonicNs());
          ++swaps;
          ++ui_only_swaps;
          last_idle_redraw_ns = idle_now_ns;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      break;
    }
    const uint64_t acquire_end = monotonicNs();
    ++received;
    if (frame.stride_bytes != kStrideBytes) throw std::runtime_error("Source stride mismatch");
    if (last_frame_id != UINT64_MAX && frame.frame_id > last_frame_id + 1) {
      dropped_ids += frame.frame_id - last_frame_id - 1;
    }
    const uint64_t frame_start = monotonicNs();
    timings.source_wait_ms.push_back(elapsedMs(acquire_start, acquire_end));
    timings.source_age_ms.push_back(elapsedMs(frame.timestamp_ns, acquire_end));

    uint64_t stage_start = monotonicNs();
    window.upload(frame.pixels);
    uint64_t stage_end = monotonicNs();
    timings.texture_upload_ms.push_back(elapsedMs(stage_start, stage_end));
    stage_start = monotonicNs();
    window.draw();
    stage_end = monotonicNs();
    timings.draw_submit_ms.push_back(elapsedMs(stage_start, stage_end));
    if (ai_bridge) {
      stage_start = monotonicNs();
      if (last_ai_request_ns == 0 || frame.timestamp_ns >= last_ai_request_ns + ai_period_ns) {
        const uint64_t rate_now_ns = monotonicNs();
        const uint64_t elapsed_ns = std::max<uint64_t>(1, rate_now_ns - start_ns);
        const double input_fps = static_cast<double>(source->publishedCount()) * 1e9 /
                                 static_cast<double>(elapsed_ns);
        const double submitted_fps = static_cast<double>(submitted + 1) * 1e9 /
                                     static_cast<double>(elapsed_ns);
        ai_bridge->request(frame, input_fps, submitted_fps);
        last_ai_request_ns = frame.timestamp_ns;
      }
      ai_bridge->receiveLatest();
      stage_end = monotonicNs();
      timings.ai_link_ms.push_back(elapsedMs(stage_start, stage_end));
    }
    stage_start = monotonicNs();
    const bool dynamic_hmi_changed = options.ai_bridge &&
                                     window.loadDynamicUiFile(options.dynamic_hmi_file);
    stage_end = monotonicNs();
    if (dynamic_hmi_changed) timings.dynamic_hmi_upload_ms.push_back(elapsedMs(stage_start, stage_end));
    if (warmup_dynamic_loaded || options.overlays || options.ai_bridge) {
      stage_start = monotonicNs();
      if (!design_ui_loaded) window.drawDynamicUi();
      stage_end = monotonicNs();
      if (!design_ui_loaded) timings.dynamic_hmi_draw_ms.push_back(elapsedMs(stage_start, stage_end));
    }
    DetectionPacket packet;
    const uint64_t overlay_now_ns = monotonicNs();
    const bool have_detection = options.design_preview
                                    ? (packet = makeDesignPreviewPacket(frame.frame_id,
                                                                         frame.timestamp_ns), true)
                                    : options.demo_boxes
                                          ? demo_worker.snapshot(frame.frame_id, overlay_now_ns, packet)
                                          : ai_bridge && ai_bridge->snapshot(
                                                frame.frame_id, overlay_now_ns, packet);
    const bool focus_matches_ai = current_view != UiView::Focus ||
                                  focus_quadrant == options.road_quadrant;
    const bool live_status = have_detection && focus_matches_ai;
    if (have_detection && packet.count > 0) {
      timings.detection_age_ms.push_back(elapsedMs(packet.timestamp_ns, overlay_now_ns));
      stage_start = monotonicNs();
      if (focus_matches_ai && current_view != UiView::Spatial) window.drawDetectionBoxes(packet);
      stage_end = monotonicNs();
      timings.detection_layer_draw_ms.push_back(elapsedMs(stage_start, stage_end));
      ++box_frames_drawn;
    }
    if (design_ui_loaded) {
      window.drawVideoThumbnails(options.layout == "quad");
      stage_start = monotonicNs();
      window.drawStaticUi();
      stage_end = monotonicNs();
      timings.static_ui_draw_ms.push_back(elapsedMs(stage_start, stage_end));
    } else {
      if (live_status) window.drawLiveStatus(packet);
      if (hmi_layer) {
        stage_start = monotonicNs();
        window.drawStaticUi();
        stage_end = monotonicNs();
        timings.static_ui_draw_ms.push_back(elapsedMs(stage_start, stage_end));
      }
      if (hmi_layer) window.drawVideoThumbnails(options.layout == "quad");
      if (hmi_layer) {
        if (current_view == UiView::Spatial) window.drawSpatialOverlay(have_detection ? &packet : nullptr);
        else window.drawSpatialInset(live_status ? &packet : nullptr);
      }
    }
    stage_start = monotonicNs();
    window.swap();
    stage_end = monotonicNs();
    timings.submitted_times_ns.push_back(stage_end);
    timings.swap_times_ns.push_back(stage_end);
    timings.egl_swap_ms.push_back(elapsedMs(stage_start, stage_end));
    timings.frame_cpu_ms.push_back(elapsedMs(frame_start, stage_end));
    ++submitted;
    ++swaps;
    last_frame_id = frame.frame_id;
  }

  source->stop();
  demo_worker.stop();
  telemetry.stop();
  const uint64_t finish_ns = monotonicNs();
  const uint64_t cpu_end = processCpuNs();
  const uint64_t duration_ns = finish_ns - start_ns;
  const auto &publish_times = source->publishTimesNs();
  for (size_t i = 1; i < publish_times.size(); ++i) {
    timings.input_interval_ms.push_back(elapsedMs(publish_times[i - 1], publish_times[i]));
  }
  for (size_t i = 1; i < timings.submitted_times_ns.size(); ++i) {
    timings.submitted_interval_ms.push_back(
        elapsedMs(timings.submitted_times_ns[i - 1], timings.submitted_times_ns[i]));
  }
  for (size_t i = 1; i < timings.swap_times_ns.size(); ++i) {
    timings.swap_interval_ms.push_back(
        elapsedMs(timings.swap_times_ns[i - 1], timings.swap_times_ns[i]));
  }
  struct rusage usage {};
  getrusage(RUSAGE_SELF, &usage);
  const uint64_t rss_peak_kb = static_cast<uint64_t>(std::max<long>(0, usage.ru_maxrss));
  const uint64_t rss_end_kb = currentRssKb();
  writeSummary(options.output_json, options, *source, window, timings, duration_ns,
               cpu_end - cpu_start, rss_peak_kb, rss_end_kb, received, submitted, swaps,
               ui_only_swaps, dropped_ids,
               telemetry, options.demo_boxes ? &demo_worker : nullptr, box_frames_drawn,
               ai_bridge.get(),
               hmi_layer ? hmi_layer->prepareMs() : 0.0,
               dynamic_cast<const PgvLiveFrameSource *>(source.get()));
  return 0;
}
}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parseOptions(argc, argv);
    return run(options);
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 2;
  }
}
