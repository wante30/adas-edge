#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
constexpr int kW = 1280;
constexpr int kH = 720;
constexpr size_t kFrame = size_t(kW) * kH * 2;
constexpr size_t kHeader = 4096;
constexpr int kSlots = 3;
constexpr size_t kRingBytes = kHeader + kSlots * kFrame;

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
uint64_t load64(unsigned char *p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
void store64(unsigned char *p, uint64_t v) { std::memcpy(p, &v, 8); }
void store32(unsigned char *p, uint32_t v) { std::memcpy(p, &v, 4); }
}

int main(int argc, char **argv) {
    std::string raw, ring, stats;
    double fps = 60.0, seconds = 0.0;
    int count = 120, start = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--raw" && i + 1 < argc) raw = argv[++i];
        else if (a == "--ring" && i + 1 < argc) ring = argv[++i];
        else if (a == "--fps" && i + 1 < argc) fps = std::atof(argv[++i]);
        else if (a == "--count" && i + 1 < argc) count = std::atoi(argv[++i]);
        else if (a == "--start-frame" && i + 1 < argc) start = std::atoi(argv[++i]);
        else if (a == "--seconds" && i + 1 < argc) seconds = std::atof(argv[++i]);
        else if (a == "--stats" && i + 1 < argc) stats = argv[++i];
    }
    if (raw.empty() || ring.empty() || count <= 0 || fps <= 0) {
        std::fprintf(stderr, "usage: --raw FILE --ring FILE [--fps 60] [--count 120]\n");
        return 2;
    }
    FILE *in = std::fopen(raw.c_str(), "rb");
    if (!in) { std::perror("raw"); return 3; }
    std::fseek(in, long(size_t(start) * kFrame), SEEK_SET);
    std::vector<std::vector<unsigned char>> frames;
    frames.reserve(size_t(count));
    for (int i = 0; i < count; ++i) {
        std::vector<unsigned char> f(kFrame);
        if (std::fread(f.data(), 1, kFrame, in) != kFrame) return 3;
        frames.emplace_back(std::move(f));
    }
    std::fclose(in);
    int fd = ::open(ring.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) { std::perror("ring"); return 4; }
    if (::ftruncate(fd, off_t(kRingBytes)) != 0) { std::perror("ftruncate"); return 4; }
    auto *map = static_cast<unsigned char *>(::mmap(nullptr, kRingBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (map == MAP_FAILED) { std::perror("mmap"); return 4; }
    std::memset(map, 0, kHeader);
    std::memcpy(map, "PGVR", 4);
    store32(map + 4, 1);
    store32(map + 8, kW);
    store32(map + 12, kH);
    store32(map + 16, kW * 2);
    store32(map + 20, kSlots);
    const auto period = std::chrono::nanoseconds(int64_t(1e9 / fps));
    auto next = std::chrono::steady_clock::now();
    const auto begin = next;
    uint64_t seq = 0;
    FILE *log = stats.empty() ? nullptr : std::fopen(stats.c_str(), "a");
    std::puts("READY");
    std::fflush(stdout);
    while (seconds <= 0 || std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() < seconds) {
        if (std::chrono::steady_clock::now() < next) std::this_thread::sleep_until(next);
        const uint32_t slot = uint32_t(seq % kSlots);
        unsigned char *meta = map + 40;
        store64(meta + slot * 8, 0);           // invalidate slot before payload
        std::memcpy(map + kHeader + size_t(slot) * kFrame,
                    frames[size_t(seq % frames.size())].data(), kFrame);
        __atomic_thread_fence(__ATOMIC_RELEASE);
        store64(meta + slot * 8, seq + 1);      // slot sequence
        store64(meta + 24 + slot * 8, seq);     // frame id (0-based)
        store64(meta + 48 + slot * 8, uint64_t(now_ns()));
        store32(map + 32, slot);
        store64(map + 24, seq + 1);             // publish latest
        ++seq;
        next += period;
        auto now = std::chrono::steady_clock::now();
        if (next <= now) next = now + period;
        if (log && seq % uint64_t(std::max(1.0, fps)) == 0) {
            const double elapsed = std::chrono::duration<double>(now - begin).count();
            std::fprintf(log, "%.3f published=%llu fps=%.3f\n", elapsed,
                         (unsigned long long)seq, seq / elapsed);
            std::fflush(log);
        }
    }
    if (log) std::fclose(log);
    std::printf("DONE published=%llu\n", (unsigned long long)seq);
    return 0;
}
