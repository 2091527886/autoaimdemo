#include "Aim.hpp"
#include <algorithm>

double Aim::get_dt() {
    if (c.prop_use_dt) {
        auto now = std::chrono::high_resolution_clock::now();
        double d = std::chrono::duration<double>(now - last_dt_ts).count();
        last_dt_ts = now;
        return d;
    } else {
        return std::max(0.001, 16.0 / 1000.0);
    }
}

double Aim::anch(double h) {
    if (!c.aim_dyn_anchor) return c.y_anchor_ratio;
    double near_h = c.dist_near_h, far_h = c.dist_far_h, hys = c.aim_hys_h;
    auto upd_seg = [&](int seg, double hh){
        if (seg == 2) { if (hh < near_h - hys) seg = 1; }
        else if (seg == 0) { if (hh > far_h + hys) seg = 1; }
        else { if (hh >= near_h) seg = 2; else if (hh <= far_h) seg = 0; }
        return seg;
    };
    last_seg_ = upd_seg(last_seg_, h);
    return (last_seg_==2) ? c.y_anchor_head : c.y_anchor_body;
}

void Aim::set_seg(double h) {
    double near_h = c.dist_near_h, far_h = c.dist_far_h, hys = c.dist_hys_h;
    int seg = last_seg_;
    if (seg == 2) { if (h < near_h - hys) seg = 1; }
    else if (seg == 0) { if (h > far_h + hys) seg = 1; }
    else { if (h >= near_h) seg = 2; else if (h <= far_h) seg = 0; }
    last_seg_ = seg;
    auto setp=[&](double kx,double ky,double tc,double lim){ algo.setParams(kx,ky,tc,lim); };
    if (!c.curve_interp) {
        if (seg==2) setp(c.seg_near_kx,c.seg_near_ky,c.seg_near_tc_ms,c.seg_near_lim);
        else if (seg==1) setp(c.seg_mid_kx,c.seg_mid_ky,c.seg_mid_tc_ms,c.seg_mid_lim);
        else setp(c.seg_far_kx,c.seg_far_ky,c.seg_far_tc_ms,c.seg_far_lim);
    } else {
        double m = (far_h + near_h) * 0.5;
        auto lerp=[&](double a,double b,double t){ return a + (b-a)*t; };
        if (h <= m) {
            double t = (h - far_h)/std::max(1e-6, (m - far_h)); t = std::clamp(t, 0.0, 1.0);
            setp(lerp(c.seg_far_kx,c.seg_mid_kx,t), lerp(c.seg_far_ky,c.seg_mid_ky,t),
                 lerp(c.seg_far_tc_ms,c.seg_mid_tc_ms,t), lerp(c.seg_far_lim,c.seg_mid_lim,t));
        } else {
            double t = (h - m)/std::max(1e-6, (near_h - m)); t = std::clamp(t, 0.0, 1.0);
            setp(lerp(c.seg_mid_kx,c.seg_near_kx,t), lerp(c.seg_mid_ky,c.seg_near_ky,t),
                 lerp(c.seg_mid_tc_ms,c.seg_near_tc_ms,t), lerp(c.seg_mid_lim,c.seg_near_lim,t));
        }
    }
}

std::pair<double,double> Aim::pred_err(const Detection& d, double dt, bool enable_pred) {
    double h = (double)d.box.height;
    double ya = anch(h);
    cv::Point2d tp(d.box.x + d.box.width/2.0,
                   d.box.y + d.box.height * ya);
    auto pred_step = [&](cv::Point2d& v_lp, const cv::Point2d& last_tp, cv::Point2d& tp_io){
        double tc = std::max(0.0, c.pred_v_tc_ms) / 1000.0;
        double a = (tc>0.0) ? (dt/(tc+dt)) : 1.0;
        double inv_dt = 1.0 / std::max(1e-6, dt);
        cv::Point2d v_inst{ (tp_io.x - last_tp.x) * inv_dt, (tp_io.y - last_tp.y) * inv_dt };
        v_lp.x = v_lp.x + a * (v_inst.x - v_lp.x);
        v_lp.y = v_lp.y + a * (v_inst.y - v_lp.y);
        double lim = c.pred_v_lim_px;
        v_lp.x = std::max(-lim, std::min(lim, v_lp.x));
        v_lp.y = std::max(-lim, std::min(lim, v_lp.y));
        double ext = std::min(dt, std::max(0.0, c.pred_dt_max_ms)/1000.0);
        tp_io.x += v_lp.x * ext;
        tp_io.y += v_lp.y * ext;
    };
    if (enable_pred && c.pred_enable && last_tp_valid_) {
        pred_step(v_lp_, last_tp_, tp);
    }
    last_tp_ = tp; last_tp_valid_ = true;
    return { tp.x - ctr.x, tp.y - ctr.y };
}
