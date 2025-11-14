#pragma once
#include "pch.hpp"
#include "Log.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

class Config {
public:
    bool timing = true;
public:
    unsigned long long frame_idx = 0;
    bool log_init_verbose = true;
    int  capture_timeout_ms_dxgi = 0;
    int  cap = 0;
    bool cap_enum = false;
    int  cap_gpu = -1;
    int  cap_out = 0;
    bool cap_cursor = false;
    bool cap_border = false;
    int  cap_retry = 3;

    bool fps_lock = true;
    int  fps = 60;
public:
    std::string engine_path = "m.engine";
    std::string resolve_engine_path() const { return engine_path; }

    bool trt_swap_rb = true;
    bool trt_use_normalize = false;
    std::vector<float> trt_mean {0.f,0.f,0.f};
    std::vector<float> trt_std  {1.f,1.f,1.f};
    float trt_border_value = 114.f;

    int  lb_block_x = 32;
    int  lb_block_y = 8;
    bool lb_use_launch_bounds = true;
    int  lb_min_blocks_per_sm = 2;
    bool lb_use_occupancy = false;
    std::string lb_block_cfg_path = "blocks_default.cfg";

    bool trt_cuda_mem = false;
    bool trt_managed_mem = false;
    int  trt_input_w = 0;
    int  trt_input_h = 0;

    const int   crop_size_px = 640;
    const float conf_thresh = 0.7f;

    const int aim_key = VK_RBUTTON;
    const int test_move_key = VK_F8;

    const double y_anchor_ratio = 0.6;
    const double px_per_360     = 3600;
    const double hfov_deg       = 85.461056;

    bool   prop = true;
    double prop_k_x = 0.8;
    double prop_k_y = 0.8;
    double prop_out_limit = 360.0;
    double prop_tc_ms = 6.0;
    double prop_dead_px = 0.6;
    bool   prop_use_dt = true;

    bool   pred_enable = true;
    double pred_dt_max_ms = 7.0;
    double pred_v_tc_ms  = 12.0;
    double pred_v_lim_px = 220.0;

    double dist_near_h = 220.0;
    double dist_far_h  = 120.0;
    double dist_hys_h  = 24.0;
    bool   curve_interp = false;

    double seg_near_kx=0.32, seg_near_ky=0.34, seg_near_tc_ms=7.0, seg_near_lim=360.0;
    double seg_mid_kx =0.24, seg_mid_ky =0.26, seg_mid_tc_ms =6.5, seg_mid_lim =330.0;
    double seg_far_kx =0.18, seg_far_ky =0.20, seg_far_tc_ms =6.0, seg_far_lim =300.0;

    bool   aim_dyn_anchor = true;
    double y_anchor_head  = 0.58;
    double y_anchor_body  = 0.80;
    double aim_hys_h      = 28.0;
    bool   aim_lock_on_seg = true;

    bool   aim_single_use_pred = true;
    bool   aim_only_with_key = true;

    bool   wind_enable = true;
    int    wind_rng_seed = 0;
    double wind_step_min_px = 0.6;
    double wind_step_max_px = 7.5;
    double wind_speed_curve = 0.65;
    double wind_end_slow_px = 10.0;
    double wind_wind_k = 0.24;
    double wind_gravity_k = 0.86;
    double wind_jitter_std_px = 0.22;
    double wind_backtrack_k = 0.12;
    double wind_near_scale = 0.90;
    double wind_mid_scale  = 1.0;
    double wind_far_scale  = 1.15;
    int    wind_max_steps_per_frame = 4;
    double wind_time_budget_ms = 0.6;
    double wind_residual_carry = 0.9;
    double wind_dead_px = 0.5;
    bool   wind_target_noise_enable = true;
    double wind_target_noise_px = 0.6;
    bool   wind_use_seg_scale = true;
    double wind_out_limit_px = 340.0;

    std::vector<std::string> classes { "enemy","teammate" };
    std::vector<int>         target_ids { 0 };

    bool capture_debug = false;
    int  capture_debug_every_n = 30;
    bool log_latency = true;
    int  log_latency_every_n = 30;
    bool latency_on_capture_only = true;
    bool capture_timing = true;
    int  capture_timing_every_n = 30;
    bool capture_timing_each_frame = false;
    bool capture_timing_on_success = true;

    bool capture_flush_after_copy = false;

    bool perf_report = true;
    int  perf_report_every_n = 30;
    bool perf_each_frame = false;

    bool opt_report = true;
    bool opt_report_capture = true;
    bool opt_report_system = true;
    int  opt_report_every_n = 30;

    bool      ovl = true;
    int       ovl_a = 160;
    int       ovl_r = 0, ovl_g = 255, ovl_b = 0;
    float     ovl_thick = 2.0f;

    bool         ovl_gpu_enum = false;
    int          ovl_gpu_idx  = 1;
    std::wstring ovl_gpu_name;
    std::wstring ovl_gpu_luid;

    bool         ovl_vsync = false;
    int          ovl_max_latency = 1;
    bool         ovl_skip_empty = false;
    bool         ovl_small_window = true;
    int          ovl_win_x = 0, ovl_win_y = 0, ovl_win_w = crop_size_px, ovl_win_h = crop_size_px;

    bool ovl_cls = true;
    std::vector<int> ovl_cr { 255, 0 };
    std::vector<int> ovl_cg { 0, 255 };
    std::vector<int> ovl_cb { 0, 0 };

    int          ovl_font_px = 40;
    std::wstring ovl_font_name;

    int ovl_only = -1;

    // crosshair and lock style
    bool  cross_show = true;
    int   cross_size = 10;      // px, half-length
    float cross_th   = 2.0f;    // px
    int   cross_a = 255, cross_r = 255, cross_g = 255, cross_b = 255;

    float lock_th = 2.0f;       // px
    int   lock_a = 255, lock_r = 255, lock_g = 215, lock_b = 0; // gold
    bool  lock_show_text = true;
    std::wstring lock_text = L"locked";

    bool enable_thread_mem_prio = false;

    int dxgi_gpu_prio = 5;
    int dxgi_max_latency = 0;

    bool is_target(int cid) const { return std::find(target_ids.begin(), target_ids.end(), cid) != target_ids.end(); }
    std::string name_or_id(int cid) const { if (cid >= 0 && cid < (int)classes.size() && !classes[cid].empty()) return classes[cid]; return std::string("id=") + std::to_string(cid); }

    void validate() {
        v_basic(); v_overlay(); v_dxgi();
        dist_near_h = std::max(1.0, dist_near_h);
        dist_far_h  = std::max(1.0, std::min(dist_near_h-1.0, dist_far_h));
        dist_hys_h  = std::max(0.0, dist_hys_h);
        aim_hys_h   = std::max(0.0, aim_hys_h);
        pred_dt_max_ms = std::max(0.0, pred_dt_max_ms);
        pred_v_tc_ms  = std::max(0.0, pred_v_tc_ms);
        pred_v_lim_px = std::max(0.0, pred_v_lim_px);
        if (!(conf_thresh >= 0.0f && conf_thresh <= 1.0f)) { lg::e()<< P_E "conf_thresh out of [0,1]"<<lg::endl; }
        if (!(y_anchor_ratio >= 0.0 && y_anchor_ratio <= 1.0)) { lg::e()<< P_E "y_anchor_ratio out of [0,1]"<<lg::endl; }
    }
private:
    void v_basic() {
        fps = std::max(1, fps);
        capture_timeout_ms_dxgi = std::max(0, capture_timeout_ms_dxgi);
        cap_retry = std::max(0, cap_retry);
        log_latency_every_n = std::max(1, log_latency_every_n);
        capture_timing_every_n = std::max(1, capture_timing_every_n);
        perf_report_every_n = std::max(1, perf_report_every_n);
        opt_report_every_n = std::max(1, opt_report_every_n);
        cap = (cap==1)?1:0;
        cap_out = std::max(0, cap_out);
        if (trt_mean.size()!=3) trt_mean = {0.f,0.f,0.f};
        if (trt_std.size()!=3)  trt_std  = {1.f,1.f,1.f};
        lb_block_x = std::max(1, lb_block_x);
        lb_block_y = std::max(1, lb_block_y);
        lb_min_blocks_per_sm = std::max(1, lb_min_blocks_per_sm);
    }
    void v_overlay() {
        if (ovl_cls) { size_t n = classes.size(); pad(ovl_cr,n,ovl_r); pad(ovl_cg,n,ovl_g); pad(ovl_cb,n,ovl_b); }
        cross_size = std::max(1, cross_size);
        cross_th   = std::max(0.5f, cross_th);
        lock_th    = std::max(1.0f, lock_th);
        auto clampc = [](int& v){ v = std::max(0, std::min(255, v)); };
        clampc(cross_a); clampc(cross_r); clampc(cross_g); clampc(cross_b);
        clampc(lock_a);  clampc(lock_r);  clampc(lock_g);  clampc(lock_b);
    }
    void v_dxgi() { dxgi_max_latency = std::max(1, dxgi_max_latency); dxgi_gpu_prio = std::min(7, std::max(-1, dxgi_gpu_prio)); }
    static void pad(std::vector<int>& v, size_t n, int d) { if (v.size()<n) v.resize(n,d); for (auto& x : v) x = std::max(0, std::min(255, x)); }
};
