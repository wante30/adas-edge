#include <cstddef>
#include <rga/im2d.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

namespace {
constexpr int kSourceWidth = 640;
constexpr int kSourceHeight = 360;
constexpr int kOutputWidth = 512;
constexpr int kOutputHeight = 288;
constexpr int kWarmupCount = 10;
constexpr int kRunCount = 200;

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const size_t index = std::min(values.size() - 1,
                                  static_cast<size_t>(std::ceil(fraction * values.size()) - 1));
    return values[index];
}
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: rga_bgr_resize_bench INPUT_BGR_RAW OUTPUT_RGB_RAW\n";
        return 2;
    }

    std::ifstream input(argv[1], std::ios::binary);
    std::vector<uint8_t> source((std::istreambuf_iterator<char>(input)),
                                std::istreambuf_iterator<char>());
    if (source.size() != static_cast<size_t>(kSourceWidth * kSourceHeight * 3)) {
        std::cerr << "unexpected BGR input size: " << source.size() << "\n";
        return 2;
    }

    std::vector<uint8_t> output(kOutputWidth * kOutputHeight * 3);
    const rga_buffer_t src = wrapbuffer_virtualaddr_t(
        source.data(), kSourceWidth, kSourceHeight,
        kSourceWidth, kSourceHeight, RK_FORMAT_BGR_888);
    const rga_buffer_t dst = wrapbuffer_virtualaddr_t(
        output.data(), kOutputWidth, kOutputHeight,
        kOutputWidth, kOutputHeight, RK_FORMAT_RGB_888);
    const rga_buffer_t pattern{};
    const im_rect srcRect{0, 0, kSourceWidth, kSourceHeight};
    const im_rect dstRect{0, 0, kOutputWidth, kOutputHeight};
    const im_rect patternRect{};

    auto process = [&]() {
        return improcess(src, dst, pattern, srcRect, dstRect, patternRect,
                         -1, nullptr, nullptr, IM_SYNC);
    };

    IM_STATUS status = IM_STATUS_FAILED;
    for (int i = 0; i < kWarmupCount; ++i) {
        status = process();
        if (status <= 0) {
            std::cerr << "RGA warmup failed: " << status << "\n";
            return 1;
        }
    }

    std::vector<double> elapsedMs;
    elapsedMs.reserve(kRunCount);
    for (int i = 0; i < kRunCount; ++i) {
        const auto started = std::chrono::steady_clock::now();
        status = process();
        const auto ended = std::chrono::steady_clock::now();
        if (status <= 0) {
            std::cerr << "RGA run failed: " << status << "\n";
            return 1;
        }
        elapsedMs.push_back(std::chrono::duration<double, std::milli>(ended - started).count());
    }

    std::ofstream result(argv[2], std::ios::binary | std::ios::trunc);
    result.write(reinterpret_cast<const char *>(output.data()),
                 static_cast<std::streamsize>(output.size()));
    if (!result) {
        std::cerr << "could not write output buffer\n";
        return 1;
    }

    const double mean = std::accumulate(elapsedMs.begin(), elapsedMs.end(), 0.0) /
                        static_cast<double>(elapsedMs.size());
    std::cout << "status=" << status << " n=" << elapsedMs.size()
              << " mean_ms=" << mean
              << " p50_ms=" << percentile(elapsedMs, 0.50)
              << " p95_ms=" << percentile(elapsedMs, 0.95)
              << " max_ms=" << *std::max_element(elapsedMs.begin(), elapsedMs.end())
              << "\n";
    return 0;
}
