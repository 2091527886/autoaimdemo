#pragma once
#include <chrono>
#include <thread>
#include <algorithm>
#include "Config.hpp"

// FPS 锁定/补偿控制（保持原名与逻辑不变）
struct FpsCtl {
    double tgt_ms = 8.0;
    size_t drop = 0;
    double ovr = 0.0;
    std::chrono::steady_clock::time_point next_tp{};
    std::chrono::steady_clock::duration per{};
    explicit FpsCtl(const Config& cfg);
    double end();
};
