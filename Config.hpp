#pragma once

#include <windows.h>
#include <avrt.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include "Log.hpp"
// Legacy alias macros removed
// Fields

// Methods


class Config { // NOTE: keep names short & readable
public:
    // 计时CSV/帧号
    bool lat_csv = true;
    std::string lat_csv_path = "latency.csv";
    unsigned long long frame_idx = 0;
    // ---------- 基础/日志 ----------
    bool log_init_verbose = true;           // 初始化阶段日志
    int  capture_timeout_ms = 16;           // AcquireNextFrame 超时（ms）

    // ---------- 可视化 ----------
    const bool enable_visual = true;
    const std::string window_title = "YOLO Real-time Detection";

    // ---------- 帧率控制 ----------
    bool fps_lock = true;                   // 启用帧率锁
    int  fps = 60;                         // 目标 FPS
    long long fps_spin_us = 0;              // 微自旋补偿（us）

    // ---------- 亲和/环境（SetThreadAffinityMask） ----------
    bool     enable_affinity = false;       // 启用线程亲和
    uint64_t affinity_main_mask = 0;        // 主线程掩码
    uint64_t affinity_overlay_mask = 0;     // Overlay 线程掩码

    // 从环境变量覆盖（可选）：AIMAFF_MAIN, AIMAFF_OVERLAY
    void load_affinity_env() {
        const std::string main_env = get_env_str("AIMAFF_MAIN");
        if (!main_env.empty()) parse_aff_env(main_env.c_str(), affinity_main_mask);
        const std::string overlay_env = get_env_str("AIMAFF_OVERLAY");
        if (!overlay_env.empty()) parse_aff_env(overlay_env.c_str(), affinity_overlay_mask);
    }
    // 预留统一入口（当前仅亲和）
    void fromEnv() { load_affinity_env(); }

private:
    static std::string get_env_str(const char* name) {
#ifdef _WIN32
        char* buf = nullptr; size_t len = 0;
        if (_dupenv_s(&buf, &len, name) == 0 && buf) { std::string v(buf); free(buf); return v; }
        return std::string();
#else
        const char* v = std::getenv(name);
        return v ? std::string(v) : std::string();
#endif
    }
    static void parse_aff_env(const char* s, uint64_t& outMask) {
        outMask = 0;
        while (*s==' '||*s=='\t') ++s;
        if (s[0]=='0' && (s[1]=='x'||s[1]=='X')) { outMask = strtoull(s, nullptr, 16); return; }
        uint64_t mask = 0; unsigned v = 0; bool innum=false;
        for (const char* p=s; ; ++p) {
            char c = *p;
            if ((c>='0'&&c<='9')) { v = v*10 + (c-'0'); innum=true; continue; }
            if (c==',' || c==' ' || c=='\t' || c=='\0') {
                if (innum) { if (v<64) mask |= (1ull<<v); v=0; innum=false; }
                if (c=='\0') break; else continue;
            }
        }
        outMask = mask;
    }

public:
    // ---------- 引擎/路径 ----------
    std::string engine_path = "l.engine";
    std::string resolve_engine_path() const { return engine_path; }

    // ---------- 截屏/检测区域 ----------
    const int   crop_size_px   = 640;       // 居中正方形
    const float conf_thresh    = 0.6f;      // 检测阈值

    // ---------- 操作/热键 ----------
    const int aim_key       = VK_RBUTTON;   // 按住平滑
    const int test_move_key = VK_F8;        // 单次移动

    // ---------- 瞄准/视角 ----------
    const double y_anchor_ratio = 0.6;      // 目标Y锚定比例
    const double px_per_360     = 3000;     // 鼠标360像素
    const double hfov_deg       = 75.0;     // 水平FOV

    // ---------- PID（像素域） ----------
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

    // ---------- 类别/标签 ----------
    std::vector<std::string> classes { "enemy","teammate" };
    std::vector<int>         target_ids { 0 };

    // ---------- 运行时日志/统计 ----------
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

    // ---------- 优化启用报告 ----------
    bool opt_report = true;
    bool opt_report_capture = true;
    bool opt_report_system = true;
    bool opt_report_verbose = true;
    int  opt_report_every_n = 30;

    // ---------- Overlay 游戏内绘制 ----------
    bool      ovl = true;              // 默认启用
    int       ovl_fps = 60;           // 刷新上限
    int       ovl_a = 160;             // 颜色 ARGB
    int       ovl_r = 0;
    int       ovl_g = 255;
    int       ovl_b = 0;
    float     ovl_thick = 2.0f;

    // ---------- Overlay GPU 选择/枚举（仅用于 Game_overlay） ----------
    bool         ovl_gpu_enum = false;        // 枚举并日志打印适配器
    int          ovl_gpu_idx  = 1;           // 0-based；<0 表示未设置
    std::wstring ovl_gpu_name;                // 名称包含匹配（优先级最高）
    std::wstring ovl_gpu_luid;                // 形如 hi:lo（十进制或0x..）

    // ---------- Overlay 同步/延迟（仅用于 Game_overlay） ----------
    bool         ovl_vsync = false;           // 默认不使用 VSync（跨适配器降低延迟）
    int          ovl_max_latency = 1;         // 最大帧延迟(1..3)，<=0 表示不控制

    // ---------- Overlay 空闲节流（仅用于 Game_overlay） ----------
    bool         ovl_skip_empty = true;       // 无绘制命令时跳过渲染与Present
    int          ovl_idle_sleep_ms = 50;      // 空闲时休眠毫秒数

    // ---------- Overlay 小窗口（仅用于 Game_overlay） ----------
    bool         ovl_small_window = true;     // 启用小窗口，默认使用裁剪尺寸
    int          ovl_win_x = 0;               // 0 表示默认居中
    int          ovl_win_y = 0;               // 0 表示默认居中
    int          ovl_win_w = crop_size_px;    // 默认用检测裁剪尺寸
    int          ovl_win_h = crop_size_px;    // 默认用检测裁剪尺寸

    // ---------- Overlay per-class 颜色（默认 enemy=红, teammate=绿） ----------
    bool ovl_cls = true;
    std::vector<int> ovl_cr { 255, 0 };
    std::vector<int> ovl_cg { 0, 255 };
    std::vector<int> ovl_cb { 0, 0 };

    // 仅显示某一类（-1 显示全部；>=0 仅该 class_id）
    int ovl_only = -1;

    // ---------- Overlay 动画（强制 DWM 产帧） ----------
    bool      overlay_anim = false;
    long long overlay_interval_100ns = 110; // 定时器间隔（100ns）
    long long overlay_spin_us = 2;          // 自旋补偿（us）
    int       overlay_px = 2;               // 像素大小
    int       overlay_x = 0, overlay_y = 0; // 位置

    // ---------- ByteTrack/Kalman 跟踪 ----------
    bool  trk = true;            // 总开关（线程）
    bool  trk_bt = true;         // 是否启用 ByteTrack（否则直通）
    bool  trk_kf = true;         // 是否启用 Kalman（预测/门控/状态输出）
    float trk_th = 0.6f;         // 高阈值
    float trk_th_lo = 0.1f;      // 低阈值
    float trk_iou = 0.6f;        // IoU 阈值（更严格，减少错配）
    int   trk_max_age = 30;      // 轨迹最大丢失帧数
    int   trk_min_hits = 3;      // 确认命中次数
    bool  trk_pred_move = false;  // 丢失时是否按预测移动（默认关闭，减少飘）
    bool  trk_cls_match = true;  // 关联是否要求类别一致
    double trk_maha = 9.49;      // 马氏距离门限（4DOF 95%≈9.49）
    bool  trk_out_nms = true;    // 输出前进行轨迹级NMS
    float trk_out_nms_iou = 0.7f; // 输出NMS IoU 阈值

    // ---------- 线程调度（MMCSS/内存优先级） ----------
    bool mmcss_main = false;
    std::wstring mmcss_task = L"Pro Audio";
    AVRT_PRIORITY mmcss_prio = AVRT_PRIORITY_HIGH;
    bool enable_thread_mem_prio = false;

    // ---------- DXGI 设备优先级/延迟 ----------
    int dxgi_gpu_prio = 3;                  // -1..7
    int dxgi_max_latency = 2;               // 1..3 推荐

    // ---------- 辅助 ----------
    bool is_target(int cid) const {
        return std::find(target_ids.begin(), target_ids.end(), cid) != target_ids.end();
    }
    std::string name_or_id(int cid) const {
        if (cid >= 0 && cid < (int)classes.size() && !classes[cid].empty()) return classes[cid];
        return std::string("id=") + std::to_string(cid);
    }

    // 约束/规范化（最小侵入）
    void validate() {
        fps = std::max(1, fps);
        capture_timeout_ms = std::max(1, capture_timeout_ms);
        log_latency_every_n = std::max(1, log_latency_every_n);
        capture_timing_every_n = std::max(1, capture_timing_every_n);
        perf_report_every_n = std::max(1, perf_report_every_n);
        opt_report_every_n = std::max(1, opt_report_every_n);
        dxgi_max_latency = std::max(1, dxgi_max_latency);
        dxgi_gpu_prio = std::min(7, std::max(-1, dxgi_gpu_prio));
        overlay_interval_100ns = std::max(1ll, overlay_interval_100ns);
        overlay_spin_us = std::max(0ll, overlay_spin_us);
        overlay_px = std::max(1, overlay_px);
        // overlay per-class 配色尺寸/范围
        auto pad = [&](std::vector<int>& v, size_t n, int d){ if (v.size()<n) v.resize(n, d); for (auto& x: v){ x = std::max(0, std::min(255, x)); } };
        if (ovl_cls) {
            size_t n = classes.size();
            pad(ovl_cr, n, ovl_r);
            pad(ovl_cg, n, ovl_g);
            pad(ovl_cb, n, ovl_b);
        }
        // 跟踪参数夹紧
        trk_th     = std::clamp(trk_th, 0.0f, 1.0f);
        trk_th_lo  = std::clamp(trk_th_lo, 0.0f, trk_th);
        trk_iou    = std::clamp(trk_iou, 0.0f, 1.0f);
        trk_max_age = std::max(1, trk_max_age);
        trk_min_hits = std::max(0, trk_min_hits);
        trk_maha = std::max(0.0, trk_maha);
        trk_out_nms_iou = std::clamp(trk_out_nms_iou, 0.0f, 1.0f);
        // 合理范围软约束（不改 const 值，仅在此处校验）
        if (!(conf_thresh >= 0.0f && conf_thresh <= 1.0f)) {
            lg::e() << P_E "conf_thresh out of [0,1]" << lg::endl;
        }
        if (!(y_anchor_ratio >= 0.0 && y_anchor_ratio <= 1.0)) {
            lg::e() << P_E "y_anchor_ratio out of [0,1]" << lg::endl;
        }
    }
};
