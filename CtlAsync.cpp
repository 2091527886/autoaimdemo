#include "CtlAsync.hpp"
#include <cmath>
#include <limits>

static Detection* pick_by_tid(std::vector<Detection>& ds, int tid) {
    if (tid < 0) return nullptr;
    for (auto& d : ds) if (d.track_id == tid) return &d;
    return nullptr;
}

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
    if (!ev_) ev_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    th_ = std::thread(&CtlAsync::run, this);
}

void CtlAsync::stop() {
    bool ex = true;
    if (go_.compare_exchange_strong(ex, false)) {
        if (ev_) SetEvent(ev_);
        if (th_.joinable()) th_.join();
        if (ev_) { CloseHandle(ev_); ev_ = nullptr; }
    }
}

void CtlAsync::put(const std::vector<Detection>& ds, bool smooth, bool single, double dt) {
    std::lock_guard<std::mutex> lk(mx_);
    msg_.ds = ds;
    msg_.smooth = smooth;
    msg_.single = single;
    msg_.dt = dt;
    seq_.fetch_add(1, std::memory_order_release);
    if (ev_) SetEvent(ev_);
}

void CtlAsync::run() {
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
        Msg cur;
        {
            std::lock_guard<std::mutex> lk(mx_);
            cur = msg_;
        }

        // 选择目标（优先沿用 track_id）
        auto ds_copy = cur.ds; // 拷贝以供 pick 修改
        Detection* target = pick_by_tid(ds_copy, cur_tid_);
        if (!target) target = pick_target(ds_copy, c, ctr_);
        if (target && target->track_id >= 0) cur_tid_ = target->track_id;

        // 无目标时：按策略决定是否移动到预测
        if (!target) {
            px.reset(); py.reset();
            if (!cur.smooth) continue; // 非平滑直接跳过
            if (!c.trk_pred_move) continue; // 配置禁止预测移动
            // 如果存在任何预测框（is_pred=true 且为目标类），选择最近中心
            Detection* pred_tgt = nullptr;
            double min_d = std::numeric_limits<double>::infinity();
            for (auto& d : ds_copy) {
                if (!d.is_pred) continue;
                if (!c.is_target(d.class_id)) continue;
                cv::Point box_center(d.box.x + d.box.width/2, d.box.y + (int)(d.box.height * c.y_anchor_ratio));
                double dist = std::hypot(box_center.x - ctr_.x, box_center.y - ctr_.y);
                if (dist < min_d) { min_d = dist; pred_tgt = &d; }
            }
            if (!pred_tgt) continue;
            // 使用预测目标进行 PID
            cv::Point tp(
                pred_tgt->box.x + pred_tgt->box.width / 2,
                pred_tgt->box.y + static_cast<int>(pred_tgt->box.height * c.y_anchor_ratio)
            );
            double err_x = (double)(tp.x - ctr_.x);
            double err_y = (double)(tp.y - ctr_.y);
            double out_x_pix = px.step(err_x, cur.dt);
            double out_y_pix = py.step(err_y, cur.dt);
            int move_x = (int)(out_x_pix * mfac_);
            int move_y = (int)(out_y_pix * mfac_);
            if (move_x || move_y) mou_.put(move_x, move_y);
            continue;
        }

        if (!cur.smooth) { px.reset(); py.reset(); }

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
