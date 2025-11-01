#include "DetAsync.hpp"
#include <chrono>

void DetAsync::start() {
    bool expected = false;
    if (!go.compare_exchange_strong(expected, true)) return;
    th = std::thread(&DetAsync::run, this);
}

void DetAsync::stop() {
    bool expected = true;
    if (go.compare_exchange_strong(expected, false)) {
        if (th.joinable()) th.join();
    }
}

void DetAsync::put(const cv::Mat& img) {
    if (img.empty()) return;
    img.copyTo(inb[wr_in]);
    in_last.store(wr_in, std::memory_order_release);
    wr_in ^= 1;
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
    using namespace std::chrono_literals;
    while (go.load(std::memory_order_relaxed)) {
        int i = in_last.load(std::memory_order_acquire);
        if (i < 0 || i == in_seen) {
            std::this_thread::sleep_for(1ms);
            continue;
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
