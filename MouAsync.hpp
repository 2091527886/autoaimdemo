#pragma once

#include <atomic>
#include <thread>
#include <chrono>
#include <windows.h>
#include "Mouse.hpp"

class Config; // fwd

// Minimal async mouse sender: keeps latest dx,dy; executes in worker thread
class MouAsync {
public:
    explicit MouAsync(const Config& cfg) : cfg_(&cfg) {}
    ~MouAsync() { stop(); }

    MouAsync(const MouAsync&) = delete;
    MouAsync& operator=(const MouAsync&) = delete;

    void start();
    void stop();

    // overwrite with latest rel move
    void put(int dx, int dy);

    // last driver call cost (ms)
    double last_ms() const { return last_ms_.load(std::memory_order_relaxed); }

private:
    void run();

    const Config* cfg_ = nullptr;

    // mailbox
    std::atomic<int> dx_{0};
    std::atomic<int> dy_{0};
    std::atomic<int> seq_{0};

    // stats
    std::atomic<double> last_ms_{0.0};

    // thread
    std::atomic<bool> go_{false};
    std::thread th_;
    HANDLE ev_ = nullptr;
};
