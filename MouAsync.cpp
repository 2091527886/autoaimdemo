#include "MouAsync.hpp"

void MouAsync::start() {
    bool expected = false;
    if (!go_.compare_exchange_strong(expected, true)) return;
    th_ = std::thread(&MouAsync::run, this);
}

void MouAsync::stop() {
    bool expected = true;
    if (go_.compare_exchange_strong(expected, false)) {
        if (th_.joinable()) th_.join();
    }
}

void MouAsync::put(int dx, int dy) {
    dx_.store(dx, std::memory_order_relaxed);
    dy_.store(dy, std::memory_order_relaxed);
    seq_.fetch_add(1, std::memory_order_release);
}

void MouAsync::run() {
    Mouse m; // lives in worker thread
    int seen = seq_.load(std::memory_order_acquire);
    while (go_.load(std::memory_order_relaxed)) {
        int s = seq_.load(std::memory_order_acquire);
        if (s == seen) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        seen = s;
        int dx = dx_.load(std::memory_order_relaxed);
        int dy = dy_.load(std::memory_order_relaxed);
        auto t0 = std::chrono::high_resolution_clock::now();
        m.Move(dx, dy);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        last_ms_.store(ms, std::memory_order_relaxed);
    }
}
