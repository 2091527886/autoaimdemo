#pragma once
#include "pch.hpp"
#include "sh.hpp"
#include "Log.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include "HC_Random.hpp"

class Config {
public:
    hc::CurveRandConfig to_hc_curve_rand_config() const { return human_rand_cfg; }

public:
    // HumanCursor 新配置（紧随类开始定义，便于查找）
    bool  human_enable = true;     // 默认开启 HumanCursor
    bool  human_steady = false;    // 稳定曲线（更小边界/更强扰动）
    int   human_queue_cap = 2048;  // 曲线步长队列上限
    int   human_mode = 0;          // 0=OffsetCurve(配合ADRC), 1=AbsoluteCurve(直达目标)
    int   human_queue_clear_policy = 0; // 0=KeepExisting(保持旧队列), 1=AbsNewClear(AbsoluteCurve 生成新曲线前清空)
    // 曲线参数配置（将 Python 随机范围/权重暴露出来）
    // AbsoluteCurve 模式的默认时长（毫秒），用于将时长映射为 target_points（不 sleep，不改计时）
    int human_abs_dur_ms_min = 200;
    int human_abs_dur_ms_max = 300;
    // 近目标极小步长抑制阈值（像素 L1），默认开启，用于抑制末端微抖；仅在 OffsetCurve 路径使用
    double human_min_step_eps = 1; // px
    bool   human_min_step_eps_enable = true;
    hc::CurveRandConfig human_rand_cfg = []{
        hc::CurveRandConfig r;
        // 强力止抖：目标附近尽量直线，几乎无扰动
        r.off_r0_min = 0; r.off_r0_max = 10;
        r.off_r1_min = 10; r.off_r1_max = 15;
        r.off_r2_min = 15; r.off_r2_max = 20;
        r.off_w0 = 100; r.off_w1 = 0; r.off_w2 = 0; // 全力压小半径
        // 完全关闭扰动
        r.mean_min = 0; r.mean_max = 0;
        r.std_min  = 0; r.std_max  = 0;
        r.freq_min = 0; r.freq_max = 0;
        // 每帧一步或极少步
        r.target_points_cap = 2;
        r.viewport_min_ratio = 0.15; r.viewport_max_ratio = 0.85;
        r.use_default_tween_pool = true;
        return r;
    }();

public:
    bool timing = true;
public:
    // ==================== [Deadzone] ====================
    bool   dz = true;       // 误差死区开关
    double dz_px = 3.0;     // 半径 px
    double dz_hys = 2;    // 滞回 px（无状态实现下仅用于阈值带）

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
    bool   adrc_enable = true;             // ADRC 总开关：true=使用 ADRC；false=不输出移动
    bool   adrc_use_measured_dt = false;  // ADRC 内部 dt 源：true=使用 Aim::get_dt() 的“实测 dt”；false=强制 16ms（锁60稳节拍更抗抖）。实测 dt 更灵敏但易抖；16ms 更稳但略钝。
    int    adrc_ctrl_low_level = 2;       // 控制律简化级别（近端抑制）：0=完整非线性；1=Fhan 简化；2=线性反馈（最稳）。建议从 2 起步，稳后再尝试 1/0。
    bool   adrc_invert_output_sign = true; // 输出方向翻转：若“越纠越远”则置 true；若“越纠越近但过度”则保持 false 并用 out_limit/beta1/2 限制速度。
    double adrc_output_limit_px = 150023.0;    // 输出限幅(px)：太慢→增大；过冲→减小（60FPS/3060 实用起步 120~240，>300 易冲，<80 易钝）
    double adrc_td_bandwidth_r = 0.5;      // TD 带宽（小误差平滑区）：大→更刚/抖动小；小→更灵/跟手快。常用 8~60；抖动大→增大；跟手慢→减小。
    double adrc_model_gain_b0 = 8;         // 被控对象增益估计：越大→补偿项越小（更“软”）；越小→补偿项越大（更“猛”）。常用 1.5~4.0；过冲→略增；偏钝→略减。
    double adrc_fal_delta = 0.02;         // fal 小量阈值：误差很小时的“软区间”宽度。常用 0.03~0.08；抖动→略增；迟滞/钝→略减（保持默认 0.05 即可）。
    double adrc_eso_beta01 = 133121;         // ESO 带宽1：越大越敏感/快，越小越稳/慢。起步可与 beta02/03 等值（如 20/20/20），按抖动/速度整体微调。
    double adrc_eso_beta02 = 1123123;          // ESO 带宽2：与 beta01/03 一起形成 ESO 的整体带宽。抖动大→整体减小；响应慢→整体增大（优先等比例调整）。
    double adrc_eso_beta03 = 150.4;          // ESO 带宽3：与 beta01/02 协同，保持三者数量级一致更容易稳定；必要时按比例整体调整。
    double adrc_nlsef_beta1 = 20.0;        // 线性反馈增益（位置 e1）：越大越快、易过冲；越小越稳、易变钝。建议与 beta2 维持 10:1～30:1 比例（如 60/6, 40/4），优先调 beta1。
    double adrc_nlsef_beta2 = 12.0;         // 线性反馈增益（速度 e2）：越大越快/易过冲；越小越稳。幅度不宜过大，常与 beta1 保持 10:1～30:1 的量级关系。
    double adrc_fal_alpha1 = 0.999;         // fal 近端指数（alpha1<1）：越小近端越“软”（抗抖）；越大近端越“硬”。常用 0.4~0.9；抖动→降；迟滞→升。
    double adrc_fal_alpha2 = 1.001;          // fal 远端指数（alpha2>1）：越大远端越“钝”（缓和）；越小远端越“猛”。常用 1.6~2.2；过冲→升；跟手慢→降。
    double adrc_fhan_c = 1.5;               // ctrl_low=1（fhan 简化）耦合系数 c，默认 1.5（更稳）。可根据手感微调。

    // ==================== [OC-SORT Tracker] ====================
    bool  ocs_enable = true;
    int   ocs_max_age = 130;
    int   ocs_min_hits = 2;
    float ocs_iou_th = 0.05f;
    float ocs_inertia = 0.0f; // VDC weight
    float ocs_kf_q_pos = 0.0f;
    float ocs_kf_q_size = 0.0f;
    float ocs_kf_r_pos = 1e-6f;
    float ocs_kf_r_size = 1e-6f;
    bool  ocs_use_byte = true;
    float ocs_byte_iou_th = 0.05f; // BYTE 二阶段匹配 IOU 阈值（官方常用较低阈值）
    std::string ocs_asso = "ciou"; // iou|giou|ciou|diou|ct_dist
    bool  ocs_oru = true;   // Observation-centric Re-Update
    bool  ocs_ocm = true;   // Observation-Centric Momentum
    bool  ocs_ocr = true;   // Observation-Centric Recovery
    float ocs_recover_relax = 0.1f; // 1.0=no relax; <1.0 relax iou_th for recovery
    int   ocs_auction_rounds = 128;   // CUDA Auction 迭代回合上限
    float ocs_ocr_slope = 0.05f;     // OCR 动态阈值斜率
    bool  ocs_disable_kf = true;     // 关闭KF预测/更新，使用观测直通（默认开启）
    float ocs_oru_alpha = 1.0f;     // ORU 融合系数（0=不用ORU，>0与obs融合）

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

    // ==================== [Aim Control] ====================
    // 仅保留 ADRC 与非风路径残差累积
    // 非风路径量化残差的保留系数（0~1），1.0=完全保留；<1.0=逐步衰减
    double aim_residual_carry = 1.0;

    bool   aim_only_with_key = true;

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
    // 死区绘制（可视化）。仅影响显示，不参与控制逻辑。
    bool  dz_show = true;
    float dz_th = 1.5f;
    int   dz_a = 160, dz_r = 255, dz_g = 255, dz_b = 255;
    int   dz_layer = 235; // 略低于 cross(240)
    int   dz_shape = 0;   // 0=box, 1=circle, 2=diamond
    int   dz_segments = 16; // circle segments


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
    // 调试中心点绘制层（默认略低于 cross）
    int       ovl_layer_dbg      = 238;
    bool      dbg_centers        = true;

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

    // center crosshair (screen center), independent of Aim target cross
    bool  cross_ctr_show = true;
    int   cross_ctr_size = 10;      // px, half-length
    float cross_ctr_th   = 2.0f;    // px
    int   cross_ctr_a = 255, cross_ctr_r = 255, cross_ctr_g = 255, cross_ctr_b = 255;

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
        // removed legacy P-branch notes
        if (false /* keep user config: do not force ADRC when P disabled */) {
            /* keep user choice: do not force ADRC here */
        }

        if (!(conf_thresh >= 0.0f && conf_thresh <= 1.0f)) { lg::e()<< P_E "conf_thresh out of [0,1]"<<lg::endl; }
        if (!(y_anchor_ratio >= 0.0 && y_anchor_ratio <= 1.0)) { lg::e()<< P_E "y_anchor_ratio out of [0,1]"<<lg::endl; }
        // 新增字段的 clamp
        human_queue_cap = MX(1, human_queue_cap);
        human_mode = (human_mode==1)?1:0;
        human_queue_clear_policy = (human_queue_clear_policy==1)?1:0;
        // clamp human_rand_cfg 部分字段
        human_rand_cfg.off_r0_min = MX(0, human_rand_cfg.off_r0_min);
        human_rand_cfg.off_r0_max = MX(human_rand_cfg.off_r0_min, human_rand_cfg.off_r0_max);
        human_rand_cfg.off_r1_min = MX(0, human_rand_cfg.off_r1_min);
        human_rand_cfg.off_r1_max = MX(human_rand_cfg.off_r1_min, human_rand_cfg.off_r1_max);
        human_rand_cfg.off_r2_min = MX(0, human_rand_cfg.off_r2_min);
        human_rand_cfg.off_r2_max = MX(human_rand_cfg.off_r2_min, human_rand_cfg.off_r2_max);
        human_rand_cfg.off_w0 = MX(0, human_rand_cfg.off_w0);
        human_rand_cfg.off_w1 = MX(0, human_rand_cfg.off_w1);
        human_rand_cfg.off_w2 = MX(0, human_rand_cfg.off_w2);
        human_rand_cfg.mean_min = MX(0, human_rand_cfg.mean_min);
        human_rand_cfg.mean_max = MX(human_rand_cfg.mean_min, human_rand_cfg.mean_max);
        human_rand_cfg.std_min = MX(0, human_rand_cfg.std_min);
        human_rand_cfg.std_max = MX(human_rand_cfg.std_min, human_rand_cfg.std_max);
        human_rand_cfg.freq_min = MX(0, human_rand_cfg.freq_min);
        human_rand_cfg.freq_max = MX(human_rand_cfg.freq_min, human_rand_cfg.freq_max);
        human_rand_cfg.viewport_min_ratio = CLP(human_rand_cfg.viewport_min_ratio, 0.0, 1.0);
        human_rand_cfg.viewport_max_ratio = CLP(human_rand_cfg.viewport_max_ratio, 0.0, 1.0);
        if (human_rand_cfg.viewport_max_ratio < human_rand_cfg.viewport_min_ratio) std::swap(human_rand_cfg.viewport_min_ratio, human_rand_cfg.viewport_max_ratio);
        human_rand_cfg.target_points_cap = MX(2, human_rand_cfg.target_points_cap);

        adrc_ctrl_low_level = std::max(0, std::min(2, adrc_ctrl_low_level));
        // DZ clamp
        dz_px = MX(0.0, dz_px);
        dz_hys = CLP(dz_hys, 0.0, dz_px);

        adrc_output_limit_px = MX(0.0, adrc_output_limit_px);
        // ADRC 数值安全护栏（避免除 0 与 pow(0,·) 等数值问题）
        adrc_td_bandwidth_r = MX(1e-6, adrc_td_bandwidth_r);
        adrc_model_gain_b0  = MX(1e-6, adrc_model_gain_b0);
        adrc_fal_delta      = MX(1e-6, adrc_fal_delta);
        adrc_fal_alpha1 = CLP(adrc_fal_alpha1, 1e-6, 0.999999);
        adrc_fal_alpha2 = MX(1.0 + 1e-6, adrc_fal_alpha2);
        adrc_fhan_c = MX(1e-6, adrc_fhan_c);
        aim_residual_carry = CLP(aim_residual_carry, 0.0, 1.0);
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
        // dz layer clamp
        clampL(dz_layer);
        dz_th = MX(0.5f, dz_th);
        auto clampc = [](int& v){ v = std::max(0, std::min(255, v)); };
        clampc(dz_a); clampc(dz_r); clampc(dz_g); clampc(dz_b);
        dz_shape = std::max(0, std::min(2, dz_shape));
        dz_segments = MX(3, dz_segments);

        clampL(ovl_layer_lock);
        clampL(ovl_layer_dbg);


        if (ovl_cls) { size_t n = classes.size(); pad(ovl_cr,n,ovl_r); pad(ovl_cg,n,ovl_g); pad(ovl_cb,n,ovl_b); }
        cross_size = MX(1, cross_size);
        cross_th   = MX(0.5f, cross_th);
        lock_th    = MX(1.0f, lock_th);
        /* reuse clampc defined above for dz */
        clampc(cross_a); clampc(cross_r); clampc(cross_g); clampc(cross_b);
        clampc(lock_a);  clampc(lock_r);  clampc(lock_g);  clampc(lock_b);

        // center crosshair clamp
        cross_ctr_size = MX(1, cross_ctr_size);
        cross_ctr_th   = MX(0.5f, cross_ctr_th);
        clampc(cross_ctr_a); clampc(cross_ctr_r); clampc(cross_ctr_g); clampc(cross_ctr_b);
    }
    void v_dxgi() { dxgi_max_latency = std::max(1, dxgi_max_latency); dxgi_gpu_prio = std::min(7, std::max(-1, dxgi_gpu_prio)); }
    static void pad(std::vector<int>& v, size_t n, int d) { if (v.size()<n) v.resize(n,d); for (auto& x : v) x = std::max(0, std::min(255, x)); }
};
