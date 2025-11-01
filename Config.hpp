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


class Config {
public:
    // ---------- 基础/日志 ----------
    bool log_init_verbose = true;           // 初始化阶段日志
    int  capture_timeout_ms = 16;           // AcquireNextFrame 超时（ms）

    // ---------- 可视化 ----------
    const bool enable_visual = true;
    const std::string window_title = "YOLO Real-time Detection";

    // ---------- 帧率控制 ----------
    bool fps_lock = false;                   // 启用帧率锁
    int  fps = 70;                         // 目标 FPS
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
    std::string engine_path = "n.engine";
    std::string resolve_engine_path() const { return engine_path; }

    // ---------- 截屏/检测区域 ----------
    const int   crop_size_px   = 640;       // 居中正方形
    const int   lock_radius_px = 320;       // 中心锁定半径
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
    int  log_latency_every_n = 60;
    bool latency_on_capture_only = true;
    bool capture_timing = true;             // 采集分步计时
    int  capture_timing_every_n = 60;
    bool capture_timing_each_frame = false;
    bool capture_timing_on_success = true;
    bool perf_report = true;                // TRT-YOLO 报告
    int  perf_report_every_n = 60;
    bool perf_each_frame = false;

    // ---------- 优化启用报告 ----------
    bool opt_report = true;
    bool opt_report_capture = true;
    bool opt_report_system = true;
    bool opt_report_verbose = true;
    int  opt_report_every_n = 60;

    // ---------- Overlay 动画（强制 DWM 产帧） ----------
    bool      overlay_anim = false;
    long long overlay_interval_100ns = 110; // 定时器间隔（100ns）
    long long overlay_spin_us = 2;          // 自旋补偿（us）
    int       overlay_px = 2;               // 像素大小
    int       overlay_x = 0, overlay_y = 0; // 位置

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
        // 合理范围软约束（不改 const 值，仅在此处校验）
        // conf_thresh/y_anchor_ratio 等 const 值仅检测
        if (!(conf_thresh >= 0.0f && conf_thresh <= 1.0f)) {
            lg::e() << P_E "conf_thresh out of [0,1]" << lg::endl;
        }
        if (!(y_anchor_ratio >= 0.0 && y_anchor_ratio <= 1.0)) {
            lg::e() << P_E "y_anchor_ratio out of [0,1]" << lg::endl;
        }
        if (lock_radius_px > crop_size_px) {
            lg::e() << P_E "lock_radius_px > crop_size_px" << lg::endl;
        }
    }
};
