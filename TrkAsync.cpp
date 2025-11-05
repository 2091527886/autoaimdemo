#include "TrkAsync.hpp"
#include <chrono>
#include "SystemTuner.hpp"

void TrkAsync::start() {
    bool ex = false;
    if (!go.compare_exchange_strong(ex, true)) return;
    if (!ev_) ev_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    th = std::thread(&TrkAsync::run, this);
}

void TrkAsync::stop() {
    bool ex = true;
    if (go.compare_exchange_strong(ex, false)) {
        if (ev_) SetEvent(ev_);
        if (th.joinable()) th.join();
        if (ev_) { CloseHandle(ev_); ev_ = nullptr; }
    }
}

void TrkAsync::put(const std::vector<Detection>& ds) {
    std::lock_guard<std::mutex> lk(mx);
    in_ds = ds; seq.fetch_add(1, std::memory_order_release);
    if (ev_) SetEvent(ev_);
}

bool TrkAsync::get(std::vector<Detection>& out) {
    int s = out_seq.load(std::memory_order_acquire);
    if (s < 0) return false;
    std::lock_guard<std::mutex> lk(mx);
    out = out_ds;
    return true;
}

void TrkAsync::run() {
    if (c.enable_affinity && c.aff_trk) {
        setAffinity(c.aff_trk, c.opt_report && c.opt_report_system);
    }
    HANDLE hMm = MmcssEnable(c.mmcss_trk, c.mmcss_trk_task.c_str(), c.mmcss_trk_prio);
    struct MmcssGuard { HANDLE h; ~MmcssGuard(){ MmcssDisable(h);} } _mmg{hMm};

    using clock = std::chrono::high_resolution_clock;
    using namespace std::chrono_literals;
    int seen = seq.load(std::memory_order_acquire);
    while (go.load(std::memory_order_relaxed)) {
        int s = seq.load(std::memory_order_acquire);
        if (s == seen) { 
            if (ev_) WaitForSingleObject(ev_, INFINITE);
            if (!go.load(std::memory_order_relaxed)) break;
            s = seq.load(std::memory_order_acquire);
            if (s == seen) continue;
        }
        seen = s;
        std::vector<Detection> cur;
        {
            std::lock_guard<std::mutex> lk(mx);
            cur = in_ds;
        }
        std::vector<Detection> out;
        auto t0 = clock::now();
        trk.update(cur, out);
        auto t1 = clock::now();
        last_ms_.store(std::chrono::duration<double, std::milli>(t1 - t0).count(), std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mx);
            out_ds.swap(out);
            out_seq.store(seen, std::memory_order_release);
        }
    }
}
