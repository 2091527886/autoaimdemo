#include "MouAsync.hpp"
#include "SystemTuner.hpp"
#include "Config.hpp"

void MouAsync::start() {
    bool expected = false;
    if (!go_.compare_exchange_strong(expected, true)) return;
    if (!ev_) {
        ev_ = CreateEventW(nullptr, FALSE, FALSE, nullptr); // auto-reset
    }
    th_ = std::thread(&MouAsync::run, this);
}

void MouAsync::stop() {
    bool expected = true;
    if (go_.compare_exchange_strong(expected, false)) {
        if (ev_) SetEvent(ev_);
        if (th_.joinable()) th_.join();
        if (ev_) { CloseHandle(ev_); ev_ = nullptr; }
    }
}

void MouAsync::put(int dx, int dy) {
    dx_.store(dx, std::memory_order_relaxed);
    dy_.store(dy, std::memory_order_relaxed);
    put_ns_.store(now_ns(), std::memory_order_relaxed);
    seq_.fetch_add(1, std::memory_order_release);
    if (ev_) SetEvent(ev_);
}

void MouAsync::run() {
    if (cfg_ && cfg_->enable_affinity && cfg_->aff_mou) {
        setAffinity(cfg_->aff_mou, cfg_->opt_report && cfg_->opt_report_system);
    }
    HANDLE hMm = MmcssEnable(cfg_ && cfg_->mmcss_mou, cfg_->mmcss_mou_task.c_str(), cfg_->mmcss_mou_prio);
    struct MmcssGuard { HANDLE h; ~MmcssGuard(){ MmcssDisable(h);} } _mmg{hMm};
    Mouse m; // lives in worker thread
    int seen = seq_.load(std::memory_order_acquire);
    while (go_.load(std::memory_order_relaxed)) {
        int s = seq_.load(std::memory_order_acquire);
        if (s == seen) {
            if (ev_) WaitForSingleObject(ev_, INFINITE);
            if (!go_.load(std::memory_order_relaxed)) break;
            s = seq_.load(std::memory_order_acquire);
            if (s == seen) continue;
        }
        seen = s;
        int dx = dx_.load(std::memory_order_relaxed);
        int dy = dy_.load(std::memory_order_relaxed);
        if (dx == 0 && dy == 0) {
            // nothing to send; zero costs for this tick
            last_call_ms_.store(0.0, std::memory_order_relaxed);
            last_lag_ms_.store(0.0, std::memory_order_relaxed);
            continue;
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        m.Move(dx, dy);
        auto t1 = std::chrono::high_resolution_clock::now();
        double call_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        long long pns = put_ns_.load(std::memory_order_relaxed);
        double lag_ms = 0.0;
        if (pns > 0) {
            long long dns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1.time_since_epoch()).count();
            long long diff = dns - pns;
            if (diff > 0) lag_ms = (double)diff / 1e6;
        }
        last_call_ms_.store(call_ms, std::memory_order_relaxed);
        last_lag_ms_.store(lag_ms, std::memory_order_relaxed);
    }
}
