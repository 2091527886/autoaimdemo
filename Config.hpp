#pragma once

#include <windows.h>
#include <avrt.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
// Legacy alias macros removed
// Fields

// Methods


class Config {
public:
    // 基础日志控制
    bool log_init_verbose = true;                // 是否打印初始化阶段的详细日志
    int  capture_timeout_ms = 16;                // AcquireNextFrame 超时（毫秒）

    // ========== 0) 基础与可视化 ==========
    // 线程亲和与手动核心绑定（单组/SetThreadAffinityMask）
    bool enable_affinity = false;            // 总开关：是否启用线程亲和
    uint64_t affinity_main_mask = 0;         // 主线程掩码（bit 对应逻辑CPU）
    uint64_t affinity_overlay_mask = 0;      // Overlay 线程掩码

    // 帧率锁定（处理完成后补足到目标间隔）
    bool fps_lock = true;                    // 启用帧率锁
    int  fps = 120;                          // 目标 FPS
    long long fps_spin_us = 0;               // 微自旋补偿（us），0 表示不自旋

    // 从环境变量覆盖（可选）：AIMAFF_MAIN, AIMAFF_OVERLAY
    void load_affinity_env() {
        const std::string main_env = get_env_str("AIMAFF_MAIN");
        if (!main_env.empty()) {
            parse_aff_env(main_env.c_str(), affinity_main_mask);
        }
        const std::string overlay_env = get_env_str("AIMAFF_OVERLAY");
        if (!overlay_env.empty()) {
            parse_aff_env(overlay_env.c_str(), affinity_overlay_mask);
        }
    }
private:
    static std::string get_env_str(const char* name) {
#ifdef _WIN32
        char* buf = nullptr; size_t len = 0;
        if (_dupenv_s(&buf, &len, name) == 0 && buf) {
            std::string v(buf); free(buf); return v;
        }
        return std::string();
#else
        const char* v = std::getenv(name);
        return v ? std::string(v) : std::string();
#endif
    }
    static void parse_aff_env(const char* s, uint64_t& outMask) {
        outMask = 0;
        // 去空白
        while (*s==' '||*s=='\t') ++s;
        if (s[0]=='0' && (s[1]=='x'||s[1]=='X')) {
            // 十六进制
            outMask = strtoull(s, nullptr, 16);
            return;
        }
        // 逗号分隔核心编号
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
    // Engine 文件（仅通过 Config 解析路径）
    std::string engine_path = "best.engine";
    std::string resolve_engine_path() const { return engine_path; }
    // 可视化窗口
    const bool enable_visual = true;
    const std::string window_title = "YOLO Real-time Detection";

    // ========== 1) 截屏与检测区域 ==========
    const int crop_size_px = 640;                    // 截取居中正方形区域尺寸
    const int lock_radius_px = 320;     // 目标距离中心阈值（像素）
    const float conf_thresh = 0.6f;      // 检测阈值

    // ========== 2) 操作与热键 ==========
    const int aim_key = VK_RBUTTON;        // 按住平滑跟随
    const int test_move_key = VK_F8;            // 单次移动测试

    // ========== 3) 瞄准与视角参数 ==========
    const double y_anchor_ratio = 0.6;            // 目标Y锚定比例（命中头/胸）
    const double px_per_360 = 3000;      // 鼠标360度所需像素（用于校正）
    const double hfov_deg = 75.0;           // 游戏水平FOV（度）

    // ========== 4) PID（像素域） ==========
    const double pid_kp_x = 0.54;
    const double pid_ki_x = 0.001;
    const double pid_kd_x = 0.00600;
    const double pid_kp_y = 0.54;
    const double pid_ki_y = 0.001;
    const double pid_kd_y = 0.00600;
    const double pid_output_limit = 400.0;        // 每帧最大像素输出
    const double pid_integral_limit = 300.0;      // 积分限幅
    const bool   pid_use_measured_dt = true;      // 用实测 dt 计算
    const double pid_fixed_dt_ms = 16.0;          // 固定 dt（毫秒），当 pid_use_measured_dt=false 时生效

    // ========== 5) 类别/标签 ==========
    std::vector<std::string> classes = { "enemy","teammate"};
    std::vector<int> target_ids = {0};

    // ========== 6) 运行时日志/统计 ==========
    // 6.0 捕获链路调试节流
    bool capture_debug = false;           // 启用后输出精简的调试统计（节流）
    int  capture_debug_every_n = 300;     // 每 N 帧输出一次调试统计
    // 6.1 延迟统计（粗粒度：capture/detect/mouse/loop）
    bool log_latency = true;               // 打印延迟统计
    int  log_latency_every_n = 60;        // 每 N 帧打印一次
    bool latency_on_capture_only = true;         // 仅统计成功截屏的帧
    // 6.2 采集分步计时（细粒度：acquire/qi/copy/map/cvt/unmap/total）
    bool capture_timing = true;       // 启用细粒度计时
    int  capture_timing_every_n = 60;     // [CAPTURE PERF] 打印周期（帧）
    bool capture_timing_each_frame = false;      // 每帧打印（调试用）
    bool capture_timing_on_success = true;   // 仅统计成功帧
    // 6.3 TensorRT-YOLO 内部性能报告
    bool perf_report = true;
    int  perf_report_every_n = 60;        // TRTYOLO 报告频率（帧）
    bool perf_each_frame = false;

    // ========== 7) 优化启用报告（启动与周期性） ==========
    bool opt_report = true;       // 总开关：是否打印优化启用信息
    bool opt_report_capture = true;        // 打印捕获链路优化信息
    bool opt_report_system = true;         // 打印系统/调度优化信息
    bool opt_report_verbose = true;      // 详细：打印OK/failed等细节
    int  opt_report_every_n = 60;// [CAPTURE OPT] 打印周期（帧）

    // ========== 8) Overlay 动画（强制 DWM 产帧，诊断/兜底） ==========
    bool overlay_anim = true;          // 启用极小透明 overlay，周期性更新以产生脏矩形
    // 更新间隔（单位：100ns；1ms=10,000×100ns）。推荐范围：5ms(50,000) ~ 16.7ms(167,000)。
    // 默认为 16ms（160,000），避免过高频率占用CPU。
    long long overlay_interval_100ns = 110; 
    // 自旋补偿预算（单位：微秒），>0 表示在等待定时器后进行短暂自旋以提升精度。推荐 0~200us。
    long long overlay_spin_us = 2;          
    // 像素大小（单位：像素）。推荐 1 或 2。
    int  overlay_px = 2;                  
    // 屏幕坐标（单位：像素）。
    int  overlay_x = 0;                       
    int  overlay_y = 0;                       

    // ========== 9) 线程调度（MMCSS） ==========
    bool mmcss_main = true;       // 在主循环线程启用 MMCSS
    std::wstring mmcss_task = L"Pro Audio"; // 任务类别（"Games"/"Pro Audio"/"Capture"等）
    AVRT_PRIORITY mmcss_prio = AVRT_PRIORITY_HIGH; // MMCSS 线程优先级

    // ========== 9.5) 线程内存优先级 ==========
    bool enable_thread_mem_prio = true;    // 启用后将主线程 MemoryPriority 显式设为 NORMAL（保证不被降级）

    // ========== 10) DXGI 设备优先级/延迟 ==========
    // GPU 线程优先级（范围 -1..7；0=normal；>0 更高）。建议：0~3 常规，5~7 谨慎使用。
    int dxgi_gpu_prio = 7;
    // 最大帧延迟（队列的最大帧数），推荐 1~3。1 通常最低延迟但可能影响吞吐。
    int dxgi_max_latency = 1;

    // ========== 辅助 ==========
    bool is_target(int cid) const {
        return std::find(target_ids.begin(), target_ids.end(), cid) != target_ids.end();
    }
    std::string name_or_id(int cid) const {
        if (cid >= 0 && cid < (int)classes.size() && !classes[cid].empty()) return classes[cid];
        return std::string("id=") + std::to_string(cid);
    }
};
