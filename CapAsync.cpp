#include "CapAsync.hpp"
#include <chrono>
#include "Config.hpp"
#include "SystemTuner.hpp"

void CapAsync::start() {
    bool expected = false;
    if (!go.compare_exchange_strong(expected, true)) return; // already running
    if (!ev) ev = CreateEventW(nullptr, FALSE, FALSE, nullptr); // auto-reset
    th = std::thread(&CapAsync::run, this);
}

void CapAsync::stop() {
    bool expected = true;
    if (go.compare_exchange_strong(expected, false)) {
        if (th.joinable()) th.join();
        if (ev) { CloseHandle(ev); ev = nullptr; }
    }
}

void CapAsync::set_roi(const cv::Rect& r) {
    // seqlock writer: increment to odd, write, increment to even
    rseq.fetch_add(1, std::memory_order_release);
    rcur = r;
    rseq.fetch_add(1, std::memory_order_release);
}

bool CapAsync::get(cv::Mat& out) {
    int i = last.load(std::memory_order_acquire);
    if (i < 0 || i == seen) return false;
    b[i].copyTo(out);
    seen = i;
    return true;
}

bool CapAsync::wait(int timeout_ms) {
    if (!ev) return false;
    DWORD to = (timeout_ms < 0) ? INFINITE : (DWORD)timeout_ms;
    DWORD r = WaitForSingleObject(ev, to);
    return r == WAIT_OBJECT_0;
}

void CapAsync::drain() {
    cap.reporter.drain();
}

void CapAsync::run() {
    HANDLE hMm = MmcssEnable(c.mmcss_cap, c.mmcss_cap_task.c_str(), c.mmcss_cap_prio);
    struct MmcssGuard { HANDLE h; ~MmcssGuard(){ MmcssDisable(h);} } _mmg{hMm};

    // Optional small warmup sleep to let main init
    using namespace std::chrono_literals;
    while (go.load(std::memory_order_relaxed)) {
        // seqlock reader: get consistent snapshot
        cv::Rect r;
        for (;;) {
            uint32_t s1 = rseq.load(std::memory_order_acquire);
            if (s1 & 1u) continue; // writer in progress
            r = rcur; // non-atomic copy
            uint32_t s2 = rseq.load(std::memory_order_acquire);
            if (s1 == s2) break; // consistent
        }
        if (r.width <= 0 || r.height <= 0) {
            std::this_thread::sleep_for(1ms);
            continue;
        }

        double wa_ms = 0.0;
        // If WGC backend, wait for frame event to avoid polling
        if (cap.is_wgc()) {
            HANDLE fe = cap.hev();
            if (fe) {
                auto w0 = std::chrono::high_resolution_clock::now();
                // finite wait to allow graceful stop
                (void)WaitForSingleObject(fe, 5);
                auto w1 = std::chrono::high_resolution_clock::now();
                wa_ms = std::chrono::duration<double, std::milli>(w1 - w0).count();
                cap.set_wait_ms(wa_ms);
                if (!go.load(std::memory_order_relaxed)) break;
            }
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        cv::Mat tmp;
        bool ok = cap.capture(tmp, r);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cap_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        last_ms_.store(wa_ms + cap_ms, std::memory_order_relaxed);
        if (ok) {
            tmp.copyTo(b[wr]);
            last.store(wr, std::memory_order_release);
            wr ^= 1;
            if (ev) SetEvent(ev);
        } else {
            // miss: no extra sleep (DXGI/WGC). AcquireNextFrame already blocks.
        }
    }
}
