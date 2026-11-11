#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

// Presentation only: no measured speed and no feedback into tracks/control.
struct SchematicRoadMotion {
    double distance = 0.0;
    std::int64_t previousNs = 0;
    bool moving = false;
    void advance(std::int64_t nowNs, std::int64_t sourceNs) {
        const double dt = previousNs > 0 && nowNs > previousNs
            ? std::min(double(nowNs - previousNs) / 1e9, 0.12) : 0.0;
        previousNs = nowNs;
        moving = sourceNs > 0 && nowNs >= sourceNs && nowNs - sourceNs <= 500000000LL;
        // 36 is a common period for 3/4/9-unit dotted, dashed and grid patterns.
        // Six relative scene units per second is animation, never vehicle speed.
        if (moving) distance = std::fmod(distance + 6.0 * dt, 36.0);
    }
    float phase(double period) const { return float(std::fmod(distance, period)); }
};
