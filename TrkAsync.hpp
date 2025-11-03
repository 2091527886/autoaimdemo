#pragma once
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <opencv2/core.hpp>
#include <windows.h>
#include "Detection.hpp"
#include "Config.hpp"
#include "ByteTrack.hpp"

class TrkAsync {
public:
    explicit TrkAsync(const Config& cfg) : c(cfg), trk(cfg) {}
    ~TrkAsync() { stop(); }

    TrkAsync(const TrkAsync&) = delete;
    TrkAsync& operator=(const TrkAsync&) = delete;

    void start();
    void stop();

    void put(const std::vector<Detection>& ds);
    bool get(std::vector<Detection>& out);

private:
    void run();

    const Config& c;
    bt::ByteTracker trk;

    // mailbox ds
    std::atomic<int> seq{0};
    std::vector<Detection> in_ds;
    std::vector<Detection> out_ds;
    std::mutex mx;

    std::atomic<int> out_seq{-1};

    std::atomic<bool> go{false};
    std::thread th;

    HANDLE ev_{nullptr};
};
