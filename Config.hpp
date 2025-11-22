#pragma once
#include "pch.hpp"
#include "sh.hpp"
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
    // ==================== [Basic & Capture] ====================
    unsigned long long frame_idx = 0;  // 全局帧计数
    bool log_init_verbose = true;      // 初始化日志更详细
    int  capture_timeout_ms_dxgi = 0;  // DXGI捕获超时（ms）
    int  cap = 0;                      // 捕获方式（0=DXGI, 1=WGC）
    bool cap_enum = false;             // 列举可用捕获设备
    int  cap_gpu = -1;                 // 捕获所用GPU（-1=自动）
    int  cap_out = 0;                  // 捕获输出索引
    bool cap_cursor = false;           // 是否捕获鼠标指针
    bool cap_border = false;           // 是否绘制捕获边框
    int  cap_retry = 3;                // 捕获失败重试次数

    bool fps_lock = true;              // 是否锁帧
    int  fps = 60;                     // 目标帧率
public:
    // ==================== [ADRC Controller] ====================
    // 使用提示（只改这些参数就能完成90%调参）：
    // - 鼠标抖动：先增大 r(+20)、或把 adrc_ctrl_low=2，再小降 beta1(-30)/beta2(-3)
    // - 太慢不跟手：先增大 adrc_out_limit(+2)，再小增 beta1(+30)/beta2(+3)
    // - 过冲：先减小 adrc_out_limit(-2)，再小降 beta1/beta2，必要时 b0(+0.1)
    // - 方向不对：adrc_inv_sign=true
    // - 帧时不稳：adrc_use_dt=false（固定16ms），稳定后再考虑置 true 并配合 r+20 或 beta1/2 小降
    // 策略说明：
    // - dt 来源：Aim::get_dt() 始终返回“实测 dt”，不受任何控制器开关影响；fps_lock 仅影响渲染节拍，不影响 dt 计算。
    // - ADRC 开关：默认启用 ADRC，P 控制器保留但默认不生效（可日后作为备选）。
    // - adrc_use_dt：仅影响 ADRC 内部是否“使用传入 dt”；为 false 时强制使用 16ms（适配锁 60 的稳定节拍）。
    // - prop_use_dt：仅影响 P 控制器内部（如启用 P 时），不影响 get_dt 的“实测 dt 来源”。
    bool   adrc_enable = true;             // ADRC 总开关（true=使用 ADRC；false 时可选择 P 控制器）
    bool   adrc_use_dt = true;           // dt 源（ADRC内部）：true=用实测 dt；false=固定16ms（锁60更稳，抖动小）
    int    adrc_ctrl_low = 2;             // 近端抑制：0=不抑制（灵敏）/1=适中/2=强（更稳，接近目标更“刹车”）
    bool   adrc_inv_sign = true;         // 方向翻转：若“越纠越远”，置 true
    double adrc_out_limit = 500.0;          // 输出限幅(px)：太慢→增大；过冲→减小（建议范围 6~14）
    double adrc_r = 9.0;               // 小误差平滑区带宽：大→更稳更“软”，抖动小时可适当减小
    double adrc_b0 = 3;                // 模型增益估计：大→控制输出变小（估小易“猛”导致过冲）
    double adrc_delta = 0.05;            // fal 小量阈值（一般保持默认）
    double adrc_beta01 = 2.0;          // ESO 带宽1：大→更快更敏感；小→更稳。抖动时适度减小
    double adrc_beta02 = 2.0;          // ESO 带宽2：同上，配合 beta01/03 调整
    double adrc_beta03 = 2.0;          // ESO 带宽3：同上
    double adrc_beta1 = 2.0;           // 跟踪环增益1（主速率）：大→更快/易过冲；小→更稳/易变钝（步长±30）
    double adrc_beta2 = 2.0;            // 跟踪环增益2（次速率）：大→更快/易过冲；小→更稳（步长±3）
    double adrc_alpha1 = 0.1;           // fal 近端指数：小→近端更“软”（抖动时可降到 0.4）
    double adrc_alpha2 = 2.5;            // fal 远端指数：大→远端更“软”（1.6~2.2）

    // ==================== [OC-SORT Tracker] ====================
    bool  ocs_enable = true;
    int   ocs_max_age = 30;
    int   ocs_min_hits = 3;
    float ocs_iou_th = 0.3f;
    float ocs_inertia = 0.2f; // VDC weight
    float ocs_kf_q_pos = 0.01f;
    float ocs_kf_q_size = 0.01f;
    float ocs_kf_r_pos = 1.0f;
    float ocs_kf_r_size = 1.0f;
    bool  ocs_use_byte = false;
    float ocs_byte_iou_th = 0.15f; // BYTE 二阶段匹配 IOU 阈值（官方常用较低阈值）
    std::string ocs_asso = "iou"; // iou|giou|ciou|diou|ct_dist
    bool  ocs_oru = true;   // Observation-centric Re-Update
    bool  ocs_ocm = true;   // Observation-Centric Momentum
    bool  ocs_ocr = true;   // Observation-Centric Recovery
    float ocs_recover_relax = 1.0f; // 1.0=no relax; <1.0 relax iou_th for recovery
    int   ocs_auction_rounds = 64;   // CUDA Auction 迭代回合上限
    float ocs_ocr_slope = 0.05f;     // OCR 动态阈值斜率
    float ocs_oru_alpha = 0.85f;     // ORU 融合系数（0=不用ORU，>0与obs融合）

    // ==================== [Selector] ====================
    // 默认 center_win，按键按下=锁定/维持
    std::string sel_mode = "center_win"; // center_win | area_first | conf_first
    int   sel_center_win_w = 220;
    int   sel_center_win_h = 220;
    float sel_area_w = 1.0f;
    float sel_conf_w = 1.0f;
    bool  sel_keep_lock = true;
    // obs 与 ds 的几何近似匹配阈值（用于从选中目标回找姿态/检测项）
    float sel_geom_center_eps_px = 2.0f;   // 中心点距离阈值（像素）
    float sel_geom_area_rel_eps = 0.02f;   // 面积相对误差阈值 [0,1]

    // Overlay 显示控制
    bool  ovl_show_obs = true; // 原始检测框（obs）
    bool  ovl_show_trk = true; // 跟踪框（track）
    bool  ovl_show_trk_all = true; // 显示全部 track 框与 id
    bool  ovl_show_pose = true; // 独立的骨架开关（默认开启）

public:
    // ==================== [Pose (TRT YOLO PoseModel)] ====================
    // 统一以 pose 作为检测来源：bbox+关键点


    bool pose_enable = true;
    std::string pose_engine_path = "mpose.engine"; // YOLO11m-pose
    bool pose_trt_swap_rb = true;
    bool pose_trt_use_normalize = false;
    std::vector<float> pose_trt_mean {0.f,0.f,0.f};
    std::vector<float> pose_trt_std  {1.f,1.f,1.f};
    bool pose_trt_cuda_mem = false;  // 内存策略（与 det 对齐；3060 建议开 cuda_mem）
    bool pose_trt_managed_mem = false;
    int  pose_trt_input_w = 640;
    int  pose_trt_input_h = 640;
    // 已移除 det/pose IoU 匹配；不再需要 pose_match_iou_thresh
    float pose_kpt_conf_thresh = 0.05f; // kpt conf threshold
    bool  pose_aim_pref_nose = false;   // true: aim nose first, then chest; false: chest first

    // ==================== [Detection Runtime (TRT通用)] ====================

    float trt_border_value = 114.f;

    // ==================== [Model Input & Thresholds] ====================
    const int   crop_size_px = 640;
    const float conf_thresh = 0.5f;

    const int aim_key = VK_RBUTTON;
    const double y_anchor_ratio = 0.6;
    const double px_per_360     = 3600;
    const double hfov_deg       = 85.461056;

    // ==================== [Aim Control & Prediction] ====================
    // 运动学、预测、距离分段与锚点等行为控制
    // P 控制器开关（与 ADRC 互斥使用）
    bool   prop_enable = false;          // 默认关闭 P 分支；不在 validate() 中强制覆盖 ADRC 开关
    bool   prop = true;                  // P 控制器内部行为参数保留（备用），默认不开启使用
    double prop_k_x = 0.4;
    double prop_k_y = 0.4;
    double prop_out_limit = 360.0;
    double prop_tc_ms = 16.0;
    double prop_dead_px = 0;
    bool   prop_use_dt = true;            // 仅影响 P 控制器内部；不影响 Aim::get_dt() 的“实测 dt 来源”

    bool   pred_enable = false;
    double pred_dt_max_ms = 7.0;
    double pred_v_tc_ms  = 12.0;
    double pred_v_lim_px = 220.0;

    double dist_near_h = 220.0;
    double dist_far_h  = 120.0;
    double dist_hys_h  = 24.0;
    bool   curve_interp = false;

    double seg_near_kx=0.4, seg_near_ky=0.4, seg_near_tc_ms=16.0, seg_near_lim=360.0;
    double seg_mid_kx =0.4, seg_mid_ky =0.4, seg_mid_tc_ms =16.0, seg_mid_lim =330.0;
    double seg_far_kx =0.4, seg_far_ky =0.4, seg_far_tc_ms =16.0, seg_far_lim =300.0;

    bool   aim_dyn_anchor = true;
    double y_anchor_head  = 0.58;
    double y_anchor_body  = 0.80;
    double aim_hys_h      = 28.0;
    bool   aim_lock_on_seg = true;

    bool   aim_only_with_key = true;

    // ==================== [Wind (Mouse micro-movement)] ====================
    // 鼠标微扰控制（平滑/随机/风力）
    bool   wind_enable = false;          // 默认关闭风鼠标（确保稳定与可控性）；开关在 Aim::on_smooth 内分流到 mv/mv_wind
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

    // ==================== [Classes & Targets] ====================
    // pose 模型类别：0=enemy, 1=teammate
    std::vector<std::string> classes { "enemy","teammate" };
    std::vector<int>         target_ids { 0 };
    int                      track_id_filter = -1; // -1: 不限制；>=0: 仅命中特定 track id

    // ==================== [Logging & Performance] ====================
    // 捕获/延迟/性能日志开关与频率
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

    // ==================== [Overlay] ====================
    // 叠加层绘制（颜色/线宽/窗口参数/字体）
    bool      ovl = true;
    int       ovl_a = 160;
    int       ovl_r = 0, ovl_g = 255, ovl_b = 0;
    float     ovl_thick = 2.0f;
    // overlay 层配置（值越大越靠上）
    int       ovl_layer_obs      = 10;
    int       ovl_layer_pose     = 15;
    int       ovl_layer_trk_all  = 20;
    int       ovl_layer_trk_sel  = 30;
    int       ovl_layer_text     = 200;
    int       ovl_layer_cross    = 240;
    int       ovl_layer_lock     = 250;

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

    int          ovl_font_px = 20;
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

    // ==================== [System & DXGI] ====================
    // 线程优先级、DXGI 捕获延迟/优先级
    bool enable_thread_mem_prio = false;

    int dxgi_gpu_prio = 5;
    int dxgi_max_latency = 0;

    bool is_target(int cid) const { return std::find(target_ids.begin(), target_ids.end(), cid) != target_ids.end(); }
    std::string name_or_id(int cid) const { if (cid >= 0 && cid < (int)classes.size() && !classes[cid].empty()) return classes[cid]; return std::string("id=") + std::to_string(cid); }

    void validate() {
        v_basic(); v_overlay(); v_dxgi();
        // 强制忽略 P 分支：prop_enable=false 时，若用户误开 P 相关参数，仍以 ADRC 为准
        if (false /* keep user config: do not force ADRC when P disabled */) {
            /* keep user choice: do not force ADRC here */
        }

        dist_near_h = MX(1.0, dist_near_h);
        dist_far_h  = MX(1.0, MN(dist_near_h-1.0, dist_far_h));
        dist_hys_h  = MX(0.0, dist_hys_h);
        aim_hys_h   = MX(0.0, aim_hys_h);
        pred_dt_max_ms = MX(0.0, pred_dt_max_ms);
        pred_v_tc_ms  = MX(0.0, pred_v_tc_ms);
        pred_v_lim_px = MX(0.0, pred_v_lim_px);
        if (!(conf_thresh >= 0.0f && conf_thresh <= 1.0f)) { lg::e()<< P_E "conf_thresh out of [0,1]"<<lg::endl; }
        if (!(y_anchor_ratio >= 0.0 && y_anchor_ratio <= 1.0)) { lg::e()<< P_E "y_anchor_ratio out of [0,1]"<<lg::endl; }
        // 新增字段的 clamp
        adrc_ctrl_low = std::max(0, std::min(2, adrc_ctrl_low));
        adrc_out_limit = MX(0.0, adrc_out_limit);
        adrc_alpha1 = CLP(adrc_alpha1, 1e-6, 0.999999);
        adrc_alpha2 = MX(1.0 + 1e-6, adrc_alpha2);
        ocs_iou_th = (float)CLP((double)ocs_iou_th, 0.0, 1.0);
        sel_center_win_w = MX(1, sel_center_win_w);
        sel_center_win_h = MX(1, sel_center_win_h);
        sel_area_w = (float)MX(0.0, (double)sel_area_w);
        sel_conf_w = (float)MX(0.0, (double)sel_conf_w);
        ocs_byte_iou_th = (float)CLP((double)ocs_byte_iou_th, 0.0, 1.0);
        ocs_recover_relax = (float)CLP((double)ocs_recover_relax, 0.5, 1.0);
        ocs_auction_rounds = MX(1, ocs_auction_rounds);
        ocs_ocr_slope = (float)CLP((double)ocs_ocr_slope, 0.0, 1.0);

    }
private:
    void v_basic() {
        // no-op for adrc defaults; only generic clamps above

        fps = MX(1, fps);
        capture_timeout_ms_dxgi = MX(0, capture_timeout_ms_dxgi);
        cap_retry = MX(0, cap_retry);
        log_latency_every_n = MX(1, log_latency_every_n);
        capture_timing_every_n = MX(1, capture_timing_every_n);
        perf_report_every_n = MX(1, perf_report_every_n);
        opt_report_every_n = MX(1, opt_report_every_n);
        cap = (cap==1)?1:0;
        cap_out = MX(0, cap_out);
    }
    void v_overlay() {
        // clamp overlay layers to [0,1000]
        auto clampL=[&](int& v){ v=std::max(0,std::min(1000,v)); };
        clampL(ovl_layer_obs);
        clampL(ovl_layer_pose);
        clampL(ovl_layer_trk_all);
        clampL(ovl_layer_trk_sel);
        clampL(ovl_layer_text);
        clampL(ovl_layer_cross);
        clampL(ovl_layer_lock);

        if (ovl_cls) { size_t n = classes.size(); pad(ovl_cr,n,ovl_r); pad(ovl_cg,n,ovl_g); pad(ovl_cb,n,ovl_b); }
        cross_size = MX(1, cross_size);
        cross_th   = MX(0.5f, cross_th);
        lock_th    = MX(1.0f, lock_th);
        auto clampc = [](int& v){ v = std::max(0, std::min(255, v)); };
        clampc(cross_a); clampc(cross_r); clampc(cross_g); clampc(cross_b);
        clampc(lock_a);  clampc(lock_r);  clampc(lock_g);  clampc(lock_b);
    }
    void v_dxgi() { dxgi_max_latency = std::max(1, dxgi_max_latency); dxgi_gpu_prio = std::min(7, std::max(-1, dxgi_gpu_prio)); }
    static void pad(std::vector<int>& v, size_t n, int d) { if (v.size()<n) v.resize(n,d); for (auto& x : v) x = std::max(0, std::min(255, x)); }
};
