#pragma once
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>
#include <opencv2/core.hpp>
#include "Detection.hpp"
#include "Config.hpp"
#include "Game_overlay.h"

class OvlAsync {
public:
    explicit OvlAsync(const Config& cfg) : c(cfg) {}
    ~OvlAsync() { stop(); }

    OvlAsync(const OvlAsync&) = delete;
    OvlAsync& operator=(const OvlAsync&) = delete;

    void start();
    void stop();

    void put(const std::vector<Detection>& ds, const cv::Rect& crop);
    void setVisible(bool v);

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

    Game_overlay gover;
};
