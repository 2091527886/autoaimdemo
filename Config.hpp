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
    // 计时CSV/帧号
    bool lat_csv = true;
    std::string lat_csv_path = "latency.csv";
    unsigned long long frame_idx = 0;
    // 基础/日志
    bool log_init_verbose = true;           // 初始化阶段日志
    int  capture_timeout_ms_dxgi = 0;       // DXGI固定0以尽可能低延迟
    int  capture_timeout_ms_wgc = 0;        // WGC默认非阻塞（0 表示完全非阻塞）

    // 捕获/GPU
    // cap: 0=DXGI DDA, 1=WGC(Windows Graphics Capture)
    int  cap = 0;
    bool cap_enum = false;      // 枚举打印
    int  cap_gpu = -1;          // 适配器索引
    std::wstring cap_gpu_name;  // 名称包含匹配（优先）
    std::wstring cap_gpu_luid;  // 形如 hi:lo
    int  cap_out = 0;           // 输出索引
    bool cap_cursor = false;    // 是否捕获鼠标（WGC）
    bool cap_border = false;    // 是否显示边框（WGC）

    // 可视化
    const bool enable_visual = true;
    const std::string window_title = "YOLO Real-time Detection";

    // 帧率
    bool fps_lock = true;                   // 启用帧率锁
    int  fps = 60;                         // 目标 FPS
    long long fps_spin_us = 0;              // 微自旋补偿（us）

    // 亲和/环境
    bool     enable_affinity = true;       // 启用线程亲和
    // 每模块亲和掩码（0 表示不绑定）
    uint64_t aff_main = 0xC000;  // 主线程
    uint64_t aff_cap  = 0xC000;  // 捕获
    uint64_t aff_det  = 0xC000;  // 检测
    uint64_t aff_trk  = 0xC000;  // 跟踪
    uint64_t aff_ctl  = 0xC000;  // 控制
    uint64_t aff_mou  = 0xC000;  // 鼠标
    uint64_t aff_ovl  = 0xC000;  // 叠加/Overlay/OverlayAnim/Game_overlay
    uint64_t aff_lat  = 0xC000;  // CSV 写入

public:
    // 引擎/路径
    std::string engine_path = "m.engine";
    std::string resolve_engine_path() const { return engine_path; }

    // 截屏/检测区域
    const int   crop_size_px   = 640;       // 居中正方形
    const float conf_thresh    = 0.6f;      // 检测阈值

    // 操作/热键
    const int aim_key       = VK_RBUTTON;   // 按住平滑
    const int test_move_key = VK_F8;        // 单次移动

    // 瞄准/视角
    const double y_anchor_ratio = 0.6;      // 目标Y锚定比例
    const double px_per_360     = 3600;     // 每360°所需鼠标计数（由脚本计算，CS=1、PF=1、Aim=1，base≈0.1）
    const double hfov_deg       = 106.639128;     // 水平FOV（垂直80°、1920x1200 -> 水平≈106.639128°）

    // PID
    const double pid_kp_x = 0.54;
    const double pid_ki_x = 0.001;
    const double pid_kd_x = 0.00600;
    const double pid_kp_y = 0.54;
    const double pid_ki_y = 0.001;
    const double pid_kd_y = 0.00600;
    const double pid_output_limit   = 400.0; // 每帧像素输出限幅
    const double pid_integral_limit = 300.0; // 积分限幅
    const bool   pid_use_measured_dt = true; // 用实测 dt
    const double pid_fixed_dt_ms     = 16.0; // 固定 dt（ms）

    // 类别/标签
    std::vector<std::string> classes { "enemy","teammate" };
    std::vector<int>         target_ids { 0 };

    // 运行时日志/统计
    bool capture_debug = false;             // 捕获链路调试
    int  capture_debug_every_n = 30;
    bool log_latency = true;                // 延迟统计
    int  log_latency_every_n = 30;
    bool latency_on_capture_only = true;
    bool capture_timing = true;             // 采集分步计时
    int  capture_timing_every_n = 30;
    bool capture_timing_each_frame = false;
    bool capture_timing_on_success = true;
    bool perf_report = true;                // TRT-YOLO 报告
    int  perf_report_every_n = 30;
    bool perf_each_frame = false;

    // 优化启用报告
    bool opt_report = true;
    bool opt_report_capture = true;
    bool opt_report_system = true;
    bool opt_report_verbose = true;
    int  opt_report_every_n = 30;

    // Overlay 绘制
    bool      ovl = true;              // 默认启用
    int       ovl_fps = 60;           // 刷新上限
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
    bool         ovl_skip_empty = true;       // 无绘制命令时跳过渲染与Present
    int          ovl_idle_sleep_ms = 50;      // 空闲时休眠毫秒数

    // Overlay 小窗口
    bool         ovl_small_window = true;     // 启用小窗口，默认使用裁剪尺寸
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

    // Overlay 动画
    bool      overlay_anim = false;
    long long overlay_interval_100ns = 110; // 定时器间隔（100ns）
    long long overlay_spin_us = 2;          // 自旋补偿（us）
    int       overlay_px = 2;               // 像素大小
    int       overlay_x = 0, overlay_y = 0; // 位置

    // 跟踪
    bool  trk = true;            // 总开关（线程）
    bool  trk_bt = true;         // 是否启用 ByteTrack（否则直通）
    bool  trk_kf = true;         // 是否启用 Kalman（预测/门控/状态输出）
    float trk_th = 0.6f;         // 高阈值
    float trk_th_lo = 0.1f;      // 低阈值
    float trk_iou = 0.6f;        // IoU 阈值（更严格，减少错配）
    int   trk_max_age = 30;      // 轨迹最大丢失帧数
    int   trk_min_hits = 3;      // 确认命中次数
    bool  trk_pred_move = true;  // 丢失时是否按预测移动（默认关闭，减少飘）
    bool  trk_cls_match = true;  // 关联是否要求类别一致
    double trk_maha = 9.49;      // 马氏距离门限（4DOF 95%≈9.49）
    bool  trk_out_nms = true;    // 输出前进行轨迹级NMS
    float trk_out_nms_iou = 0.7f; // 输出NMS IoU 阈值

    // 线程调度
    // 主线程
    bool mmcss_main = true;
    std::wstring mmcss_task = L"Games";
    AVRT_PRIORITY mmcss_prio = AVRT_PRIORITY_HIGH;
    bool enable_thread_mem_prio = false;

    // 模块线程（按需启用，不使用环境变量；短名、可读）
    bool           mmcss_cap = true;           // 捕获
    std::wstring   mmcss_cap_task = L"Capture";
    AVRT_PRIORITY  mmcss_cap_prio = AVRT_PRIORITY_HIGH;

    bool           mmcss_det = true;           // 检测
    std::wstring   mmcss_det_task = L"Games";
    AVRT_PRIORITY  mmcss_det_prio = AVRT_PRIORITY_NORMAL;

    bool           mmcss_trk = true;           // 跟踪
    std::wstring   mmcss_trk_task = L"Games";
    AVRT_PRIORITY  mmcss_trk_prio = AVRT_PRIORITY_HIGH;

    bool           mmcss_ctl = true;           // 控制/鼠标
    std::wstring   mmcss_ctl_task = L"Pro Audio";
    AVRT_PRIORITY  mmcss_ctl_prio = AVRT_PRIORITY_HIGH;

    bool           mmcss_mou = true;           // 鼠标发送
    std::wstring   mmcss_mou_task = L"Pro Audio";
    AVRT_PRIORITY  mmcss_mou_prio = AVRT_PRIORITY_HIGH;


    bool           mmcss_ovl = true;           // 叠加绘制
    std::wstring   mmcss_ovl_task = L"Playback";
    AVRT_PRIORITY  mmcss_ovl_prio = AVRT_PRIORITY_NORMAL;

    // DXGI
   bool dxgi_wait_vblank = true;            // DXGI是否在捕获前等待VBlank（事件驱动）
   int dxgi_gpu_prio = 3;                  // -1..7
   int dxgi_max_latency = 2;               // 1..3 推荐

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
        v_trk();
        v_dxgi();
        // 合理范围软约束（不改 const 值，仅在此处校验）
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
        // WGC: allow 0 to mean fully non-blocking
        capture_timeout_ms_wgc  = std::max(0, capture_timeout_ms_wgc);
        log_latency_every_n = std::max(1, log_latency_every_n);
        capture_timing_every_n = std::max(1, capture_timing_every_n);
        perf_report_every_n = std::max(1, perf_report_every_n);
        opt_report_every_n = std::max(1, opt_report_every_n);
        overlay_interval_100ns = std::max(1ll, overlay_interval_100ns);
        overlay_spin_us = std::max(0ll, overlay_spin_us);
        overlay_px = std::max(1, overlay_px);
        // 捕获后端
        cap = (cap==1)?1:0;
        cap_out = std::max(0, cap_out);
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
    void v_trk() {
        trk_th     = std::clamp(trk_th, 0.0f, 1.0f);
        trk_th_lo  = std::clamp(trk_th_lo, 0.0f, trk_th);
        trk_iou    = std::clamp(trk_iou, 0.0f, 1.0f);
        trk_max_age = std::max(1, trk_max_age);
        trk_min_hits = std::max(0, trk_min_hits);
        trk_maha = std::max(0.0, trk_maha);
        trk_out_nms_iou = std::clamp(trk_out_nms_iou, 0.0f, 1.0f);
    }

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
