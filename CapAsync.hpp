#pragma once

#include <atomic>
#include <thread>
#include <opencv2/opencv.hpp>
#include "ScreenCapturer.hpp"
#include <windows.h>

class Config;

// Minimal async wrapper around ScreenCapturer (logic unchanged)
class CapAsync {
public:
    explicit CapAsync(const Config& cfg)
        : c(cfg), cap(cfg, cfg.crop_size_px, cfg.crop_size_px) {}

    ~CapAsync() { stop(); }

    // Non-copyable
    CapAsync(const CapAsync&) = delete;
    CapAsync& operator=(const CapAsync&) = delete;

    // Control
    void start();
    void stop();

    // Wait for new frame ready; timeout_ms<0 -> infinite, =0 -> poll
    bool wait(int timeout_ms);

    // expose backend mode and event handle for advanced waits
    bool is_wgc() const { return cap.is_wgc(); }
    HANDLE hev() const { return ev; }

    // ROI management: set once or at runtime (thread-safe)
    void set_roi(const cv::Rect& r);

    // Fetch latest frame (returns false if no new frame)
    bool get(cv::Mat& out);

    // Drain capture stats (print in main thread)
    void drain();

    // Last capture wall time (ms), includes wait/serve decisions
    double last_ms() const { return last_ms_.load(std::memory_order_relaxed); }

    // Proxy dimensions (from ScreenCapturer)
    int w() const { return cap.w(); }
    int h() const { return cap.h(); }

private:
    void run();

    const Config& c;
    ScreenCapturer cap;

    // double buffer
    cv::Mat b[2];
    int wr = 0;            // writer index (0/1)
    std::atomic<int> last{-1}; // last produced index
    int seen = -1;         // last consumed index

    // ROI state
    std::atomic<uint32_t> rseq{0};
    cv::Rect rcur{0,0,0,0};

    // thread control
    std::atomic<bool> go{false};
    std::thread th;

    // frame-ready event (auto-reset)
    HANDLE ev = nullptr;

    // ROI-set event (auto-reset)
    HANDLE ev_roi = nullptr;

    // timing
    std::atomic<double> last_ms_{0.0};
};
