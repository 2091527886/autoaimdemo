#pragma once
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>
#include <opencv2/core.hpp>
#include <windows.h>
#include "Detection.hpp"
#include "Config.hpp"
#include "Game_overlay.h"

class OvlAsync {
public:
    explicit OvlAsync(const Config& cfg) : c(cfg) {}
    ~OvlAsync() { stop(); }

    // last overlay draw cost (ms)
    double last_ms() const { return last_ms_.load(std::memory_order_relaxed); }

    OvlAsync(const OvlAsync&) = delete;
    OvlAsync& operator=(const OvlAsync&) = delete;

    void start();
    void stop();

    void put(const std::vector<Detection>& ds, const cv::Rect& crop);
private:
    void run();

    const Config& c;
    std::atomic<bool> go{false};
    std::thread th;

    // mailbox
    std::atomic<int> seq{0};
    std::vector<Detection> ds_;
    cv::Rect crop_{};
    std::mutex mx;

    HANDLE ev_{nullptr};
    Game_overlay gover;

    // stats
    std::atomic<double> last_ms_{0.0};
};

