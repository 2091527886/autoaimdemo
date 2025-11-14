#include "FpsCtl.hpp"
#include <algorithm>

FpsCtl::FpsCtl(const Config& cfg) {
    tgt_ms = 1000.0 / std::max(1, cfg.fps);
    per = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double, std::milli>(tgt_ms));
    next_tp = std::chrono::steady_clock::now() + per;
}

double FpsCtl::end() {
    using clk = std::chrono::steady_clock;
    auto now = clk::now();
    if (now < next_tp) {
        std::this_thread::sleep_until(next_tp);
        ovr = 0.0;
        next_tp += per;
        return tgt_ms;
    }
    // lagging: compute overshoot and catch up phase without sleeping
    ovr = std::chrono::duration<double, std::milli>(now - next_tp).count();
    while (next_tp <= now) { next_tp += per; ++drop; }
    return tgt_ms;
}
