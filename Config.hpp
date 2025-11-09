#pragma once

#include <windows.h>
#include <avrt.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include "Log.hpp"



class Config {
public:
public:
    // 计时总开关（关闭则尽量不做统计计时）
    bool timing = true;
public:
    // 计时CSV/帧号
    unsigned long long frame_idx = 0;
    // 基础/日志
    bool log_init_verbose = true;           // 初始化阶段日志
    int  capture_timeout_ms_dxgi = 0;       // DXGI固定0以尽可能低延迟
    // 捕获/GPU
    // cap: 0=DXGI DDA, 1=WGC(Windows Graphics Capture)
    int  cap = 0;
    bool cap_enum = false;      // 枚举打印（调试用，默认关）
    int  cap_gpu = -1;          // 适配器索引
    int  cap_out = 0;           // 输出索引
    bool cap_cursor = false;    // 是否捕获鼠标（WGC）
    bool cap_border = false;    // 是否显示边框（WGC）
    int  cap_retry = 3;         // 捕获失败立即重试次数（0=不重试）

    // 可视化
    
    

    // 帧率
    bool fps_lock = true; // 帧率锁
    int  fps = 60;      // 目标FPS（更适合高刷显示器）
    // 调度/环境（简化，无显式亲和设置）
public:
    // 引擎/路径
    std::string engine_path = "m.engine";
    std::string resolve_engine_path() const { return engine_path; }

    // TRT-YOLO 前处理/内核配置（3060 专项）
    bool trt_swap_rb = true;                 // GPU侧 BGRA->RGB
    bool trt_use_normalize = false;          // 是否启用规范化（mean/std）
    std::vector<float> trt_mean {0.f, 0.f, 0.f};
    std::vector<float> trt_std  {1.f, 1.f, 1.f};
    float trt_border_value = 114.f;          // letterbox 边界值

    // 内核块/占用/launch_bounds 配置
    // 预设（RTX 3060 / SM86 推荐）
    // - 32x8,  MINB=2: 通用稳定，寄存器压力适中，吞吐与占用平衡（默认）
    // - 16x16, MINB=2: 更均衡的二维并行分布，对部分输入高宽相近时有更好访存局部性
    // - 64x4,  MINB=2: 提高X方向访存合并度，适合宽图像；注意寄存器/指令吞吐可能上升
    // - 32x16/64x8（占用候选）: 在大分辨率且bandwidth占优时可试；当寄存器使用偏高时可能降低SM并发
    // 使用建议：
    //   0) 基线：32x8 + MINB=2；
    //   1) 核偏轻且SM空闲：试 16x16 或 64x4（仍为256线程/块）
    //   2) 若仍瓶颈：试 32x16 或 64x8（512线程/块），实测寄存器/并发；
    //   3) 若寄存器压力导致active blocks下降，降低 MINB 或退回 32x8；
    //   4) use_occupancy=1 仅探索候选，锁定后请固定 block_x/block_y，避免抖动。
    int  lb_block_x = 32;
    int  lb_block_y = 8;
    bool lb_use_launch_bounds = true;
    int  lb_min_blocks_per_sm = 2;
    bool lb_use_occupancy = false;           // 运行时基于占用率推荐块大小（探索用，最终建议固定）
    std::string lb_block_cfg_path = "blocks_default.cfg"; // 外部块配置文件路径（可选），示例键：
                                             //   block_x, block_y, use_launch_bounds, min_blocks_per_sm, use_occupancy
                                             // 示例：
                                             //   block_x=32
                                             //   block_y=8
                                             //   use_launch_bounds=1
                                             //   min_blocks_per_sm=2
                                             //   use_occupancy=0

    // 内存策略/输入尺寸（可选）
    bool trt_cuda_mem = false;               // 输入已在 CUDA 显存
    bool trt_managed_mem = false;            // 统一内存
    int  trt_input_w = 0;                    // 固定输入宽（0 表示动态）
    int  trt_input_h = 0;                    // 固定输入高（0 表示动态）

    // 截屏/检测区域
    const int   crop_size_px   = 640; // 居中正方形
    const float conf_thresh    = 0.7f;      // 检测阈值

    // 操作/热键
    const int aim_key       = VK_RBUTTON;   // 按住平滑
    const int test_move_key = VK_F8;        // 单次移动

    // 瞄准/视角
    const double y_anchor_ratio = 0.6;      // 目标Y锚定比例
    const double px_per_360     = 3600;     // 每360°所需鼠标计数（由脚本计算，CS=1、PF=1、Aim=1，base≈0.1）
    const double hfov_deg       = 85.461056;     // 水平FOV（垂直60°、1920x1200 -> 水平≈85.461056°）

    // 移除 PID（完全由 PropAlgo 替代）

    // PropAlgo（比例 + 平滑）
    bool   prop = true;                  // 默认启用
    double prop_k_x = 0.34;
    double prop_k_y = 0.34;
    double prop_out_limit = 360.0;       // 每帧像素限幅（Hipfire 预设）
    double prop_tc_ms = 6.0;             // 一阶低通时间常数（ms）
    double prop_dead_px = 0.45;           // 误差死区（像素）
    bool   prop_use_dt = true;           // 使用实测 dt

    // 预测
    bool   pred_enable = true;           // 启用短期预测
    double pred_dt_max_ms = 12.0;        // 最大外推时间（ms）
    double pred_v_tc_ms  = 18.0;         // 速度低通时间常数（ms）
    double pred_v_lim_px = 320.0;        // 速度幅度限幅（像素/秒）

    // 距离度量与分段
    double dist_near_h = 220.0;          // 高度>=近
    double dist_far_h  = 120.0;          // 高度<=远
    double dist_hys_h  = 20.0;           // 滞回
    bool   curve_interp = false;         // 分段间是否插值（方案A可关闭）

    // 分段参数（重配置 AimAlgo）
    double seg_near_kx   = 0.28, seg_near_ky   = 0.28;
    double seg_near_tc_ms= 6.0;  double seg_near_lim = 420.0;
    double seg_mid_kx    = 0.22, seg_mid_ky    = 0.22;
    double seg_mid_tc_ms = 6.5;  double seg_mid_lim  = 360.0;
    double seg_far_kx    = 0.16, seg_far_ky    = 0.16;
    double seg_far_tc_ms = 7.0;  double seg_far_lim  = 300.0;

    // 动态锚点（近头远身）
    bool   aim_dyn_anchor = true;
    double y_anchor_head  = 0.2;
    double y_anchor_body  = 0.60;
    double aim_hys_h      = 20.0;        // 锚点切换滞回
    bool   aim_lock_on_seg = true;       // 段内保持命中部位

    // 行为开关
    bool   aim_single_use_pred = true;   // 单次点击是否使用预测
    bool   aim_only_with_key = true;     // 仅在按键时启用分段/预测

    // 风鼠标（人类化轨迹）
    bool   wind_enable = false;           // 总开关（均衡型默认：开启）
    int    wind_rng_seed = 0;            // 0=随机播种，>0=固定种子

    // 步长与速度
    // 均衡型默认
    double wind_step_min_px = 0.7;       // 微步最小像素
    double wind_step_max_px = 9.0;       // 微步最大像素（上限保护）
    double wind_speed_curve = 0.55;       // 0..1，越大越早加速后期减速更明显
    double wind_end_slow_px = 12.0;      // 终端减速触发距离（像素）

    // 风/重力/抖动强度
    // 均衡型默认
    double wind_wind_k = 0.28;           // 风力系数：横向扰动
    double wind_gravity_k = 0.90;        // 重力系数：朝目标的吸引
    double wind_jitter_std_px = 0.28;    // 高斯抖动标准差
    double wind_backtrack_k = 0.10;      // 过冲回拉（小反向力）
    // 分段缩放（与已有距离分段一致，按目标框高度 h）
    // 均衡型默认
    double wind_near_scale = 0.92;        // 近距离：更稳更短步
    double wind_mid_scale  = 1.0;        // 中距离：默认
    double wind_far_scale  = 1.12;        // 远距离：更快更长步

    // 单帧预算与残差
    // 均衡型默认
    int    wind_max_steps_per_frame = 5; // 单帧最多微步次数（硬限制）
    double wind_time_budget_ms = 1.0;    // 单帧最多用时（软限制，优先于上限）
    double wind_residual_carry = 1.0;    // 残差占比（1=全额滚入下一帧）
    // 终端控制与精度
    // 均衡型默认
    double wind_dead_px = 0.45;           // 风鼠标内部死区（避免抖动）
    bool   wind_target_noise_enable = false; // 是否启用终端微抖
    double wind_target_noise_px = 0.0;   // 终端微抖（更“人类化”）

    // 与现有分段/预测的一致性控制
    bool   wind_use_seg_scale = true;    // 是否按近/中/远缩放风强度/步长
    // 兼容/安全
    double wind_out_limit_px = 450.0;    // 单帧总输出像素上限（保护）
    // 类别/标签
    std::vector<std::string> classes { "enemy","teammate" };
    std::vector<int>         target_ids { 0 };

    // 运行时日志/统计
    bool capture_debug = false;             // 捕获链路调试
    int  capture_debug_every_n = 30;
    bool log_latency = true;                // 延迟统计
    int  log_latency_every_n = 30;
    bool latency_on_capture_only = true; // 仅采集失败时也计入统计则设为false
    bool capture_timing = true;             // 采集分步计时
    int  capture_timing_every_n = 30;
    bool capture_timing_each_frame = false;
    bool capture_timing_on_success = true;
    
    // 提交：是否在 Copy 后立即 Flush（DXGI/WGC 共用）
    bool capture_flush_after_copy = false;

    bool perf_report = true;                // TRT-YOLO 报告
    int  perf_report_every_n = 30;
    bool perf_each_frame = false;

    // 优化启用报告
    bool opt_report = true;
    bool opt_report_capture = true;
    bool opt_report_system = true;
    int  opt_report_every_n = 30;

    // Overlay 绘制
    bool      ovl = true;              // 默认启用
    int       ovl_a = 160;             // 颜色 ARGB
    int       ovl_r = 0;
    int       ovl_g = 255;
    int       ovl_b = 0;
    float     ovl_thick = 2.0f;

    // Overlay GPU
    bool         ovl_gpu_enum = false;        // 枚举并日志打印适配器
    int          ovl_gpu_idx  = 1;           // 0-based；<0 表示未设置
    std::wstring ovl_gpu_name;                // 名称包含匹配（优先级最高）
    std::wstring ovl_gpu_luid;                // 形如 hi:lo（十进制或0x..）

    // Overlay 同步/延迟
    bool         ovl_vsync = false;           // 默认不使用 VSync（跨适配器降低延迟）
    int          ovl_max_latency = 1;         // 最大帧延迟(1..3)，<=0 表示不控制

    // Overlay 空闲节流
    bool         ovl_skip_empty = false;       // 无绘制命令时跳过渲染与Present

    // Overlay 小窗口
    bool         ovl_small_window = true; // 小窗口（默认使用裁剪尺寸）
    int          ovl_win_x = 0;               // 0 表示默认居中
    int          ovl_win_y = 0;               // 0 表示默认居中
    int          ovl_win_w = crop_size_px;    // 默认用检测裁剪尺寸
    int          ovl_win_h = crop_size_px;    // 默认用检测裁剪尺寸

    // Overlay 颜色
    bool ovl_cls = true;
    std::vector<int> ovl_cr { 255, 0 };
    std::vector<int> ovl_cg { 0, 255 };
    std::vector<int> ovl_cb { 0, 0 };

    // 仅显示某一类（-1 显示全部；>=0 仅该 class_id）
    int ovl_only = -1;

    // 跟踪
    // 线程调度
    // 主线程
    bool mmcss_main = false; // 默认关（当前未使用，仅保留占位）
    std::wstring mmcss_task = L"Capture";
    AVRT_PRIORITY mmcss_prio = AVRT_PRIORITY_HIGH;
    bool enable_thread_mem_prio = false;
    // 模块线程（按需启用，不使用环境变量；短名、可读）


    // DXGI
   int dxgi_gpu_prio = 5;                  // -1..7
   int dxgi_max_latency = 0;               // 1..3 推荐

    // 辅助
    bool is_target(int cid) const {
        return std::find(target_ids.begin(), target_ids.end(), cid) != target_ids.end();
    }
    std::string name_or_id(int cid) const {
        if (cid >= 0 && cid < (int)classes.size() && !classes[cid].empty()) return classes[cid];
        return std::string("id=") + std::to_string(cid);
    }

    // 约束/规范化（最小侵入）
    void validate() {
        v_basic();
        v_overlay();
        v_dxgi();
        // 合理范围软约束（不改 const 值，仅在此处校验）
        // 分段/预测校验（仅夹紧与范围提示，不改语义）
        dist_near_h = std::max(1.0, dist_near_h);
        dist_far_h  = std::max(1.0, std::min(dist_near_h-1.0, dist_far_h));
        dist_hys_h  = std::max(0.0, dist_hys_h);
        aim_hys_h   = std::max(0.0, aim_hys_h);
        pred_dt_max_ms = std::max(0.0, pred_dt_max_ms);
        pred_v_tc_ms  = std::max(0.0, pred_v_tc_ms);
        pred_v_lim_px = std::max(0.0, pred_v_lim_px);
        if (!(conf_thresh >= 0.0f && conf_thresh <= 1.0f)) {
            lg::e() << P_E "conf_thresh out of [0,1]" << lg::endl;
        }
        if (!(y_anchor_ratio >= 0.0 && y_anchor_ratio <= 1.0)) {
            lg::e() << P_E "y_anchor_ratio out of [0,1]" << lg::endl;
        }
    }

private:
    // 基础/计时/通用
    void v_basic() {
        fps = std::max(1, fps);
        capture_timeout_ms_dxgi = std::max(0, capture_timeout_ms_dxgi);
        cap_retry = std::max(0, cap_retry);
        log_latency_every_n = std::max(1, log_latency_every_n);
        capture_timing_every_n = std::max(1, capture_timing_every_n);
        perf_report_every_n = std::max(1, perf_report_every_n);
        opt_report_every_n = std::max(1, opt_report_every_n);
        // 捕获后端
        cap = (cap==1)?1:0;
        cap_out = std::max(0, cap_out);
        // 规范化参数
        if (trt_mean.size()!=3) trt_mean = {0.f,0.f,0.f};
        if (trt_std.size()!=3)  trt_std  = {1.f,1.f,1.f};
        lb_block_x = std::max(1, lb_block_x);
        lb_block_y = std::max(1, lb_block_y);
        lb_min_blocks_per_sm = std::max(1, lb_min_blocks_per_sm);
    }
    
    // Overlay 相关
    void v_overlay() {
        if (ovl_cls) {
            size_t n = classes.size();
            pad(ovl_cr, n, ovl_r);
            pad(ovl_cg, n, ovl_g);
            pad(ovl_cb, n, ovl_b);
        }
    }

    // 跟踪参数夹紧
    // DXGI 范围
    void v_dxgi() {
        dxgi_max_latency = std::max(1, dxgi_max_latency);
        dxgi_gpu_prio = std::min(7, std::max(-1, dxgi_gpu_prio));
    }

    // 向量尺寸与范围修正（0..255）
    static void pad(std::vector<int>& v, size_t n, int d) {
        if (v.size() < n) v.resize(n, d);
        for (auto& x : v) x = std::max(0, std::min(255, x));
    }
};
