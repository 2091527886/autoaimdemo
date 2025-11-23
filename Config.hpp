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
    // 概览：
    // - 本 ADRC 用于“像素误差→像素校正量”，再乘以 mfac 映射鼠标。默认优先于 P 控制器使用（adrc_enable=true）。
    // - Aim::get_dt() 始终返回实测 dt；ADRC 内部通过 adrc_use_dt 选择“使用实测 dt”或“强制 16ms（锁60稳节拍）”。
    // - 目标环境：Windows、锁 60 FPS、显卡 3060、风鼠标默认关闭，仅 C++。
    //
    // 参数分组与核心作用：
    // 1) 开关与方向：
    //    - adrc_enable: 总开关。true=使用 ADRC；false 时如 prop_enable=true 则改用 P。
    //    - adrc_inv_sign: 输出方向翻转。若“越纠越远”置 true。
    //    - adrc_use_dt: ADRC 内部 dt 源。true=用实测 dt；false=16ms（锁60更稳，抖动小）。
    //    - adrc_ctrl_low: 控制律简化级别（近端抑制）。0=完整非线性；1=Fhan 简化；2=线性反馈（最稳）。建议从 2 起步。
    // 2) 输出与约束：
    //    - adrc_out_limit: 输出限幅（px），硬限制每帧最大矫正。太小→推不动；太大→猛冲。
    //      60FPS/3060 常用起步 120~240，再依手感微调；>300 易冲，<80 易钝。
    //    - adrc_fal_delta: fal 小量阈值，决定误差很小时的“软区间”。一般 0.03~0.08，默认 0.05 即可。
    // 3) 跟踪微分器（TD）：
    //    - adrc_td_bandwidth_r: 小误差平滑区带宽。大→更刚、抖动小；小→更灵、跟手快。常用 8~60。抖动大→增大；跟手慢→减小。
    // 4) 扩张状态观测器（ESO）：
    //    - adrc_model_gain_b0: 对对象增益的估计。大→补偿项偏小（更“软”）；小→补偿项偏大（更“猛”）。常用 1.5~4.0。
    //    - adrc_eso_beta01/02/03: ESO 带宽。大→敏感/快；小→稳/慢。起步可等值（如 20/20/20），按抖动/速度整体微调。
    // 5) 非线性反馈（NLSEF）：
    //    - adrc_nlsef_beta1/adrc_nlsef_beta2: 线性反馈增益（位置/速度）。过大→快/易冲；过小→稳/钝。常用比例 10:1～30:1（如 60/6, 40/4）。
    //    - adrc_fal_alpha1/adrc_fal_alpha2: fal 指数。alpha1<1 近端更“软”；alpha2>1 远端更“钝”。常用 0.4~0.9 / 1.6~2.2。
    //
    // 调参顺序建议（锁60/3060）：
    // 1) 确认方向：adrc_inv_sign 使矫正朝向目标。
    // 2) 先限幅：adrc_out_limit 设为期望最大帧移动量的 1.5~2 倍。
    // 3) dt 决策：锁 60 且帧稳→adrc_use_dt=false；若想跟随微小帧抖→true（配合 r↑或 beta1/2↓）。
    // 4) ctrl_low：2→1→0，逐步放开；始终在“稳”的前提下提升速度。
    // 5) ESO：中等起步（20/20/20）。抖动→整体↓；慢→整体↑。
    // 6) 增益：以 beta1 为主、beta2 为辅，逐步提升到“快而不过冲”。先调限幅，再调增益。
    // 7) r：整体稳后用 r 微调“近端稳/远端快”的平衡。
    //
    // 常见症状→改法：
    // - 抖动/锯齿：r↑、beta1/2↓、alpha1↓、adrc_use_dt=false。
    // - 跟手慢：out_limit↑、beta1/2↑、r↓（少量）。
    // - 过冲：out_limit↓、beta1/2↓、b0↑（少量）、alpha2↑。
    // - 小范围来回不收敛：ctrl_low=2、r↑、beta1/2↓、alpha1↓、delta↑（小幅）。

   /*
    [ADRC 进阶说明：结构/交互/排障/预设]

    1) 结构直觉
    - TD（adrc_td_bandwidth_r）：
      目标是给“理想参考轨迹 v1/v2”。r 越大，近端越“硬”（抖动少、但有钝感），越小则“灵”（跟手快、但可能抖）。
      TD 相当于“误差微分+平滑器”，是“滤抖/速度”的第一关。

    - ESO（adrc_eso_beta01/02/03, adrc_model_gain_b0）：
      z1/z2/z3 ≈ 系统输出/速度/总扰动（含未建模部分）。
      beta 越大，估计越快越敏感；越小，越稳越慢。三者保持同量级，优先“等比例”调整。
      b0 是“受控对象增益”的估计：估小=以为对象很弱→补偿会大（猛）；估大=以为对象很强→补偿会小（软）。

    - NLSEF（adrc_nlsef_beta1/2, adrc_fal_alpha1/2）：
      beta1 对“位置误差 e1”主导速度，beta2 对“速度误差 e2”抑制震荡。
      常见配比 10:1～30:1（如 40/4、60/6），先调 beta1 决定“快慢”，再用 beta2 抑制“冲/抖”。
      alpha1<1 让近端“更软”（抗抖），alpha2>1 让远端“更钝”（防猛冲）。

    2) 交互与优先级
    - 先“方向正确”（adrc_invert_output_sign），再“限幅”（adrc_output_limit_px），再“dt 决策”（adrc_use_measured_dt），再“增益”，最后“r 微调”。
    - dt 决策：
      • false（16ms）：锁60更稳，抗抖。适合“稳重型”或帧波动环境。
      • true（实测）：更灵敏、但可能引入细小抖动；配合 r↑ 或 beta1/2↓。
    - ctrl_low_level：
      • 2（线性）最稳 → 1（fhan）→ 0（完整非线性），逐步放开。任何时候不牺牲稳定性。

    3) 决策树（简）
    - 越纠越远？→ adrc_invert_output_sign=true
    - 过冲（冲过目标/来回摆动）？→ adrc_output_limit_px↓ → adrc_nlsef_beta1/2↓ → adrc_fal_alpha2↑ → adrc_model_gain_b0↑
    - 微抖（细小锯齿）？→ adrc_td_bandwidth_r↑ → adrc_nlsef_beta1/2↓ → adrc_fal_alpha1↓ → adrc_use_measured_dt=false
    - 跟手慢（发木）？→ adrc_output_limit_px↑ → adrc_nlsef_beta1/2↑ → adrc_td_bandwidth_r↓（小幅） → adrc_model_gain_b0↓（小幅）
    - 小范围来回不收敛？→ ctrl_low_level=2 → r↑ → beta1/2↓ → alpha1↓ → fal_delta↑

    4) 经典故障模式与根因
    - “点到点抽动”：out_limit 过大 + beta1 偏大 → 降 out_limit，beta1 小步下降，alpha2 上调
    - “轻微高频振”：r 偏小 + dt 抖/ESO 过敏感 → r 上调、use_measured_dt=false 或 beta01/02/03 降
    - “大幅过冲后回摆”：beta1 偏大 + b0 偏小 → beta1 降，b0 略升，out_limit 降
    - “远端快，近端抖”：alpha1 偏大（近端太硬）→ alpha1 降；必要时 r↑ 或 beta2↑
    - “平移慢、跟步延迟”：out_limit 太小 + beta1 偏小 → out_limit↑，beta1↑；若抖增大，再用 r↑ 或 alpha1↓

    5) 样例预设（锁60/3060，默认 adrc_invert_output_sign=true）
    - 稳重型（抗抖优先，少许钝）：use_measured_dt=false, ctrl_low_level=2
      out_limit=180, r=28, b0=2.8, beta01/02/03=20/20/20, beta1/2=40/4, alpha1/2=0.6/1.9
      说明：限幅中等，ESO居中，位置环偏稳，速度环抑冲，近端偏软、远端偏钝。

    - 均衡型（稳与快平衡）：use_measured_dt=false, ctrl_low_level=2
      out_limit=220, r=32, b0=2.4, beta01/02/03=24/24/24, beta1/2=50/5, alpha1/2=0.6/1.9
      说明：适合大多数场景，微调 beta1/2 即可顺/逆时针调整快/稳。

    - 迅捷型（跟手优先，注意限冲）：use_measured_dt=true, ctrl_low_level=2
      out_limit=240, r=26, b0=2.0, beta01/02/03=28/28/28, beta1/2=60/6, alpha1/2=0.7/1.8
      说明：更灵敏；遇抖先 r↑ 或 beta1/2↓，必要时切回 use_measured_dt=false。

    6) 安全护栏
    - alpha1 必须 <1；alpha2 必须 >1。
    - out_limit 过大几乎必冲；过小必钝。
    - b0 过小 → “猛”、易冲；过大 → “软”、跟手慢。优先小幅调整（±0.2~0.4）。
    - 调参一次只动一组，配日志记录；感觉变坏立刻回滚上一步。

    7) 实测 dt 策略
    - use_measured_dt=false（锁16ms）：
      • 优点：近端几乎不受帧抖影响；缺点：对“帧间真实速度变化”略迟。
      • 推荐：帧波动明显、或偏稳手感时。
    - use_measured_dt=true（实测）：
      • 优点：更灵；缺点：易受轻微帧波影响。
      • 推荐：配合 r↑（+20%）或 beta1/2↓（-10~20%），并用 out_limit 兜底。

    8) 调参顺序（清单）
    - 步1：确认 adrc_invert_output_sign 正确（手感方向正确）。
    - 步2：设定 adrc_output_limit_px（决定“能否推得动且不猛冲”的边界）。
    - 步3：决定 dt 策略 adrc_use_measured_dt（锁16ms优先稳）。
    - 步4：ctrl_low_level=2 起步，必要时再到 1/0。
    - 步5：ESO 带宽等比例调整（从 20/20/20 起）。
    - 步6：增益 beta1/2（10~30:1），先提速，再抑冲。
    - 步7：用 r 微调“近端稳/远端快”的平衡。
   */

    bool   adrc_enable = true;             // ADRC 总开关：true=使用 ADRC；false 时如 prop_enable=true 则改用 P（P 作为备选保留）。
    bool   adrc_use_measured_dt = false;  // ADRC 内部 dt 源：true=使用 Aim::get_dt() 的“实测 dt”；false=强制 16ms（锁60稳节拍更抗抖）。实测 dt 更灵敏但易抖；16ms 更稳但略钝。
    int    adrc_ctrl_low_level = 2;       // 控制律简化级别（近端抑制）：0=完整非线性；1=Fhan 简化；2=线性反馈（最稳）。建议从 2 起步，稳后再尝试 1/0。
    bool   adrc_invert_output_sign = true; // 输出方向翻转：若“越纠越远”则置 true；若“越纠越近但过度”则保持 false 并用 out_limit/beta1/2 限制速度。
    double adrc_output_limit_px = 1500.0;    // 输出限幅(px)：太慢→增大；过冲→减小（60FPS/3060 实用起步 120~240，>300 易冲，<80 易钝）
    double adrc_td_bandwidth_r = 0.5;      // TD 带宽（小误差平滑区）：大→更刚/抖动小；小→更灵/跟手快。常用 8~60；抖动大→增大；跟手慢→减小。
    double adrc_model_gain_b0 = 8;         // 被控对象增益估计：越大→补偿项越小（更“软”）；越小→补偿项越大（更“猛”）。常用 1.5~4.0；过冲→略增；偏钝→略减。
    double adrc_fal_delta = 0.02;         // fal 小量阈值：误差很小时的“软区间”宽度。常用 0.03~0.08；抖动→略增；迟滞/钝→略减（保持默认 0.05 即可）。
    double adrc_eso_beta01 = 1;         // ESO 带宽1：越大越敏感/快，越小越稳/慢。起步可与 beta02/03 等值（如 20/20/20），按抖动/速度整体微调。
    double adrc_eso_beta02 = 1;          // ESO 带宽2：与 beta01/03 一起形成 ESO 的整体带宽。抖动大→整体减小；响应慢→整体增大（优先等比例调整）。
    double adrc_eso_beta03 = 150.4;          // ESO 带宽3：与 beta01/02 协同，保持三者数量级一致更容易稳定；必要时按比例整体调整。
    double adrc_nlsef_beta1 = 20.0;        // 线性反馈增益（位置 e1）：越大越快、易过冲；越小越稳、易变钝。建议与 beta2 维持 10:1～30:1 比例（如 60/6, 40/4），优先调 beta1。
    double adrc_nlsef_beta2 = 12.0;         // 线性反馈增益（速度 e2）：越大越快/易过冲；越小越稳。幅度不宜过大，常与 beta1 保持 10:1～30:1 的量级关系。
    double adrc_fal_alpha1 = 0.999;         // fal 近端指数（alpha1<1）：越小近端越“软”（抗抖）；越大近端越“硬”。常用 0.4~0.9；抖动→降；迟滞→升。
    double adrc_fal_alpha2 = 1.001;          // fal 远端指数（alpha2>1）：越大远端越“钝”（缓和）；越小远端越“猛”。常用 1.6~2.2；过冲→升；跟手慢→降。

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
    bool   prop = false;                  // P 控制器内部行为参数保留（备用），默认不开启使用
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
        adrc_ctrl_low_level = std::max(0, std::min(2, adrc_ctrl_low_level));
        adrc_output_limit_px = MX(0.0, adrc_output_limit_px);
        adrc_fal_alpha1 = CLP(adrc_fal_alpha1, 1e-6, 0.999999);
        adrc_fal_alpha2 = MX(1.0 + 1e-6, adrc_fal_alpha2);
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
