#include "OvlAsync.hpp"
#include <chrono>
#include <algorithm>
#include "Log.hpp"

void OvlAsync::start() {
    bool ex = false;
    if (!go.compare_exchange_strong(ex, true)) return;
    if (!ev_) ev_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    lg::o() << P_B "Overlay=ON (fps=" << c.ovl_fps
            << ", a=" << c.ovl_a << ", rgb=" << c.ovl_r << "," << c.ovl_g << "," << c.ovl_b
            << ", th=" << c.ovl_thick << ")" << lg::endl;
    th = std::thread(&OvlAsync::run, this);
}

void OvlAsync::stop() {
    bool ex = true;
    if (go.compare_exchange_strong(ex, false)) {
        if (ev_) SetEvent(ev_);
        if (th.joinable()) th.join();
        if (ev_) { CloseHandle(ev_); ev_ = nullptr; }
        lg::o() << P_B "Overlay=OFF" << lg::endl;
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
    if (ev_) SetEvent(ev_);
}

void OvlAsync::run() {
    gover.UseVirtualScreen();
    gover.SetMaxFPS((unsigned)std::max(1, c.ovl_fps));
    gover.SetConfig(c);
    {
        bool ok = gover.Start();
        lg::ok_or_werr(P_B, "Overlay.Start", ok);
        if (!ok) return;
    }

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
        cv::Rect r;
        {
            std::lock_guard<std::mutex> lk(mx);
            cur = ds_;
            r = crop_;
        }

        gover.BeginFrame();
        for (const auto& d : cur) {
            if (c.ovl_only >= 0 && d.class_id != c.ovl_only) continue;
            float x = (float)(r.x + d.box.x);
            float y = (float)(r.y + d.box.y);
            float w = (float)d.box.width;
            float h = (float)d.box.height;
            int cid = d.class_id;
            int rr = c.ovl_r, gg = c.ovl_g, bb = c.ovl_b;
            if (c.ovl_cls && cid >= 0) {
                if (cid < (int)c.ovl_cr.size()) rr = c.ovl_cr[cid];
                if (cid < (int)c.ovl_cg.size()) gg = c.ovl_cg[cid];
                if (cid < (int)c.ovl_cb.size()) bb = c.ovl_cb[cid];
            }
            auto col = ARGB((uint8_t)c.ovl_a,(uint8_t)rr,(uint8_t)gg,(uint8_t)bb);
            gover.AddRect({x,y,w,h}, col, (float)c.ovl_thick);
            if (d.track_id >= 0) {
                std::wstring txt = L"#" + std::to_wstring(d.track_id);
                gover.AddText(x, std::max(0.0f, y - 12.0f), txt, 14.0f, ARGB(255,255,255,255));
            }
        }
        gover.EndFrame();
    }

    gover.Stop();
}
