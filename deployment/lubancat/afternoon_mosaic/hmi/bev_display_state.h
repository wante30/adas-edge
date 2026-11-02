#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

// Presentation only. Never passed to tracking, risk decisions or control.
namespace BevDisplay {
constexpr double HoldSeconds = 1.0;
struct State {
    float x = 0, z = 0, tx = 0, tz = 0;
    std::int64_t measuredNs = 0;
    bool observe(float rawX, float rawZ, std::int64_t timestampNs) {
        if (timestampNs <= measuredNs || !std::isfinite(rawX) || !std::isfinite(rawZ)) return false;
        if (!measuredNs) { x = rawX; z = rawZ; }
        tx = rawX; tz = rawZ; measuredNs = timestampNs;
        return true;
    }
    void advance(double dt) {
        const float k = float(-std::expm1(-5.0 * std::clamp(dt, 0.0, 0.12)));
        x += (tx - x) * k;
        z += (tz - z) * k;
    }
    double age(std::int64_t nowNs) const { return std::max(0.0, double(nowNs - measuredNs) / 1e9); }
    bool expired(std::int64_t nowNs) const { return age(nowNs) > HoldSeconds; }
    float fade(std::int64_t nowNs) const {
        return float(std::clamp((HoldSeconds - age(nowNs)) / 0.35, 0.0, 1.0));
    }
};
}
