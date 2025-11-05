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

    // last driver call cost (ms) - kept for compatibility
    double last_ms() const { return last_call_ms_.load(std::memory_order_relaxed); }
    // end-to-end lag from put() to driver call completion (ms)
    double last_lag_ms() const { return last_lag_ms_.load(std::memory_order_relaxed); }
    // take-and-clear: read lag once, then reset to 0 to avoid sticky values across frames
    double take_lag_ms() { return last_lag_ms_.exchange(0.0, std::memory_order_relaxed); }

private:
    void run();

    static inline long long now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

    const Config* cfg_ = nullptr;

    // mailbox
    std::atomic<int> dx_{0};
    std::atomic<int> dy_{0};
    std::atomic<int> seq_{0};
    std::atomic<long long> put_ns_{0}; // timestamp of last put

    // stats
    std::atomic<double> last_call_ms_{0.0}; // driver call cost
    std::atomic<double> last_lag_ms_{0.0};  // put->done lag

    // thread
    std::atomic<bool> go_{false};
    std::thread th_;
    HANDLE ev_ = nullptr;
};
