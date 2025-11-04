#include "DetAsync.hpp"
#include <chrono>
#include "Config.hpp"
#include "SystemTuner.hpp"

void DetAsync::start() {
    bool expected = false;
    if (!go.compare_exchange_strong(expected, true)) return;
    if (!ev_in) ev_in = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    th = std::thread(&DetAsync::run, this);
}

void DetAsync::stop() {
    bool expected = true;
    if (go.compare_exchange_strong(expected, false)) {
        if (ev_in) SetEvent(ev_in);
        if (th.joinable()) th.join();
        if (ev_in) { CloseHandle(ev_in); ev_in = nullptr; }
    }
}

void DetAsync::put(const cv::Mat& img) {
    if (img.empty()) return;
    img.copyTo(inb[wr_in]);
    in_last.store(wr_in, std::memory_order_release);
    wr_in ^= 1;
    if (ev_in) SetEvent(ev_in);
}

bool DetAsync::get(std::vector<Detection>& out, Timing& tm) {
    int i = out_last.load(std::memory_order_acquire);
    if (i < 0 || i == out_seen) return false;
    out = outb[i];
    tm = tmb[i];
    out_seen = i;
    return true;
}

void DetAsync::run() {
    HANDLE hMm = MmcssEnable(c.mmcss_det, c.mmcss_det_task.c_str(), c.mmcss_det_prio);
    struct MmcssGuard { HANDLE h; ~MmcssGuard(){ MmcssDisable(h);} } _mmg{hMm};

    using namespace std::chrono_literals;
    while (go.load(std::memory_order_relaxed)) {
        int i = in_last.load(std::memory_order_acquire);
        if (i < 0 || i == in_seen) {
            if (ev_in) WaitForSingleObject(ev_in, INFINITE);
            if (!go.load(std::memory_order_relaxed)) break;
            i = in_last.load(std::memory_order_acquire);
            if (i < 0 || i == in_seen) continue;
        }
        // work on a local copy to keep producer free
        cv::Mat img;
        inb[i].copyTo(img);
        in_seen = i;

        // detect
        std::vector<Detection> ds;
        Timing tt;
        od.detect(img, ds, c, tt);

        // publish
        outb[wr_out] = std::move(ds);
        tmb[wr_out] = tt;
        out_last.store(wr_out, std::memory_order_release);
        wr_out ^= 1;
    }
}
