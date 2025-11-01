#pragma once

#include <atomic>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Detection.hpp"
#include "Timing.hpp"
#include "ObjDet.hpp"

class Config;

// Minimal async wrapper around ObjDet (logic unchanged: always process latest)
class DetAsync {
public:
    explicit DetAsync(const Config& cfg)
        : c(cfg), od(cfg) {}
    ~DetAsync() { stop(); }

    DetAsync(const DetAsync&) = delete;
    DetAsync& operator=(const DetAsync&) = delete;

    void start();
    void stop();

    // Push latest frame (copy); mailbox semantics overwrite older
    void put(const cv::Mat& img);

    // Fetch latest detection result; returns false if no new result
    bool get(std::vector<Detection>& out, Timing& tm);

private:
    void run();

    const Config& c;
    ObjDet od; // lives in detection thread only

    // input mailbox (double buffer)
    cv::Mat inb[2];
    int wr_in = 0;
    std::atomic<int> in_last{-1};
    int in_seen = -1;

    // output mailbox (double buffer)
    std::vector<Detection> outb[2];
    Timing tmb[2];
    int wr_out = 0;
    std::atomic<int> out_last{-1};
    int out_seen = -1;

    // thread control
    std::atomic<bool> go{false};
    std::thread th;
};
