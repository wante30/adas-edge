#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
constexpr int kW = 1280;
constexpr int kH = 720;
constexpr size_t kBytes = size_t(kW) * kH * 2;
constexpr size_t kHeader = 40;

void put32(unsigned char *p, uint32_t v) {
    p[0] = v & 0xff; p[1] = (v >> 8) & 0xff; p[2] = (v >> 16) & 0xff; p[3] = v >> 24;
}
void put64(unsigned char *p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (v >> (8 * i)) & 0xff;
}

int64_t ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}

int main(int argc, char **argv) {
    std::string raw, output, stats;
    double fps = 60.0, seconds = 0.0;
    int count = 120, start = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--raw" && i + 1 < argc) raw = argv[++i];
        else if (a == "--output" && i + 1 < argc) output = argv[++i];
        else if (a == "--fps" && i + 1 < argc) fps = std::atof(argv[++i]);
        else if (a == "--count" && i + 1 < argc) count = std::atoi(argv[++i]);
        else if (a == "--start-frame" && i + 1 < argc) start = std::atoi(argv[++i]);
        else if (a == "--seconds" && i + 1 < argc) seconds = std::atof(argv[++i]);
        else if (a == "--stats" && i + 1 < argc) stats = argv[++i];
    }
    if (raw.empty() || output.empty() || count <= 0 || fps <= 0) {
        std::fprintf(stderr, "usage: --raw FILE --output FILE [--fps 60] [--count 120] [--seconds N]\n");
        return 2;
    }
    FILE *in = std::fopen(raw.c_str(), "rb");
    if (!in) { std::perror("raw"); return 3; }
    std::fseek(in, static_cast<long>(size_t(start) * kBytes), SEEK_SET);
    std::vector<std::vector<unsigned char>> frames;
    frames.reserve(size_t(count));
    for (int i = 0; i < count; ++i) {
        std::vector<unsigned char> f(kBytes);
        if (std::fread(f.data(), 1, kBytes, in) != kBytes) {
            std::fprintf(stderr, "short raw frame %d\n", i); return 3;
        }
        frames.emplace_back(std::move(f));
    }
    std::fclose(in);

    int outfd = ::open(output.c_str(), O_RDWR | O_CREAT, 0644);
    if (outfd < 0) { std::perror("output"); return 4; }
    if (::ftruncate(outfd, off_t(kHeader + kBytes)) != 0) { std::perror("ftruncate"); return 4; }
    const auto period = std::chrono::nanoseconds(static_cast<int64_t>(1000000000.0 / fps));
    const auto startTime = std::chrono::steady_clock::now();
    auto next = startTime;
    uint32_t published = 0;
    FILE *log = stats.empty() ? nullptr : std::fopen(stats.c_str(), "a");
    double last = 0;
    std::puts("READY");
    std::fflush(stdout);
    std::vector<unsigned char> record(kHeader + kBytes);
    while (seconds <= 0 || std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count() < seconds) {
        if (std::chrono::steady_clock::now() < next)
            std::this_thread::sleep_until(next);
        const uint32_t source = published % uint32_t(frames.size());
        const uint32_t generation = published / uint32_t(frames.size());
        std::memcpy(record.data(), "PGVF", 4);
        put32(record.data() + 4, 1);
        put32(record.data() + 8, kW);
        put32(record.data() + 12, kH);
        put32(record.data() + 16, published);
        put32(record.data() + 20, generation);
        put32(record.data() + 24, 0);
        put32(record.data() + 28, uint32_t(kBytes));
        put64(record.data() + 32, uint64_t(ns()));
        std::memcpy(record.data() + kHeader, frames[source].data(), kBytes);
        if (::pwrite(outfd, record.data() + kHeader, kBytes, off_t(kHeader)) != ssize_t(kBytes)) {
            std::perror("pwrite payload"); return 4;
        }
        if (::pwrite(outfd, record.data(), kHeader, 0) != ssize_t(kHeader)) {
            std::perror("pwrite header"); return 4;
        }
        ++published;
        next += period;
        auto now = std::chrono::steady_clock::now();
        if (next <= now) next = now + period;
        double elapsed = std::chrono::duration<double>(now - startTime).count();
        if (log && elapsed - last >= 1.0) {
            std::fprintf(log, "%.3f published=%u fps=%.3f\n", elapsed, published, published / elapsed);
            std::fflush(log);
            last = elapsed;
        }
    }
    if (log) std::fclose(log);
    ::close(outfd);
    std::printf("DONE published=%u fps=%.3f\n", published,
                published / std::max(0.001, std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - startTime).count()));
    return 0;
}
