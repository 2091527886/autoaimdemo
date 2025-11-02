#include "CtlAsync.hpp"
#include <cmath>
#include <limits>

static Detection* pick_target(std::vector<Detection>& ds, const Config& c, const cv::Point& ctr) {
    Detection* target = nullptr;
    double min_dist_to_center = std::numeric_limits<double>::infinity();
    for (auto& det : ds) {
        if (!c.is_target(det.class_id)) continue;
        cv::Point box_center(det.box.x + det.box.width / 2, det.box.y + det.box.height / 2);
        double dist = std::hypot(box_center.x - ctr.x, box_center.y - ctr.y);
        if (dist < min_dist_to_center) { min_dist_to_center = dist; target = &det; }
    }
    return target;
}

void CtlAsync::start() {
    bool ex = false;
    if (!go_.compare_exchange_strong(ex, true)) return;
    th_ = std::thread(&CtlAsync::run, this);
}

void CtlAsync::stop() {
    bool ex = true;
    if (go_.compare_exchange_strong(ex, false)) {
        if (th_.joinable()) th_.join();
    }
}

void CtlAsync::put(const std::vector<Detection>& ds, bool smooth, bool single, double dt) {
    std::lock_guard<std::mutex> lk(mx_);
    msg_.ds = ds;
    msg_.smooth = smooth;
    msg_.single = single;
    msg_.dt = dt;
    seq_.fetch_add(1, std::memory_order_release);
}

void CtlAsync::run() {
    int seen = seq_.load(std::memory_order_acquire);
    while (go_.load(std::memory_order_relaxed)) {
        int s = seq_.load(std::memory_order_acquire);
        if (s == seen) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        seen = s;
        Msg cur;
        {
            std::lock_guard<std::mutex> lk(mx_);
            cur = msg_;
        }

        // 选择目标
        auto ds_copy = cur.ds; // 拷贝以供 pick 修改
        Detection* target = pick_target(ds_copy, c, ctr_);
        if (!cur.smooth || !target) {
            px.reset(); py.reset();
            if (!target) continue; // 单发在下一段处理
        }

        if (cur.smooth && target) {
            cv::Point tp(
                target->box.x + target->box.width / 2,
                target->box.y + static_cast<int>(target->box.height * c.y_anchor_ratio)
            );
            double err_x = (double)(tp.x - ctr_.x);
            double err_y = (double)(tp.y - ctr_.y);
            double out_x_pix = px.step(err_x, cur.dt);
            double out_y_pix = py.step(err_y, cur.dt);
            int move_x = (int)(out_x_pix * mfac_);
            int move_y = (int)(out_y_pix * mfac_);
            if (move_x || move_y) mou_.put(move_x, move_y);
        }

        if (cur.single && target) {
            cv::Point tp(
                target->box.x + target->box.width / 2,
                target->box.y + static_cast<int>(target->box.height * c.y_anchor_ratio)
            );
            double dx = (double)(tp.x - ctr_.x);
            double dy = (double)(tp.y - ctr_.y);
            int move_x = (int)(dx * mfac_);
            int move_y = (int)(dy * mfac_);
            if (move_x || move_y) mou_.put(move_x, move_y);
        }
    }
}
