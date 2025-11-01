#include "OvlAsync.hpp"
#include <chrono>
#include <algorithm>

void OvlAsync::start() {
    bool ex = false;
    if (!go.compare_exchange_strong(ex, true)) return;
    th = std::thread(&OvlAsync::run, this);
}

void OvlAsync::stop() {
    bool ex = true;
    if (go.compare_exchange_strong(ex, false)) {
        if (th.joinable()) th.join();
    }
}

void OvlAsync::setVisible(bool v) {
    gover.SetVisible(v);
}

void OvlAsync::put(const std::vector<Detection>& ds, const cv::Rect& crop) {
    std::lock_guard<std::mutex> lk(mx);
    ds_ = ds;
    crop_ = crop;
    seq.fetch_add(1, std::memory_order_release);
}

void OvlAsync::run() {
    gover.UseVirtualScreen();
    gover.SetMaxFPS((unsigned)std::max(1, c.ovl_fps));
    gover.Start();

    int seen = seq.load(std::memory_order_acquire);

    while (go.load(std::memory_order_relaxed)) {
        int s = seq.load(std::memory_order_acquire);
        if (s == seen) {
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
            continue;
        }
        seen = s;

        std::vector<Detection> cur;
        cv::Rect r;
        {
            std::lock_guard<std::mutex> lk(mx);
            cur = ds_;
            r = crop_;
        }

        gover.BeginFrame();
        for (const auto& d : cur) {
            float x = (float)(r.x + d.box.x);
            float y = (float)(r.y + d.box.y);
            float w = (float)d.box.width;
            float h = (float)d.box.height;
            gover.AddRect({x,y,w,h}, ARGB((uint8_t)c.ovl_a,(uint8_t)c.ovl_r,(uint8_t)c.ovl_g,(uint8_t)c.ovl_b), (float)c.ovl_thick);
        }
        gover.EndFrame();
    }

    gover.Stop();
}
