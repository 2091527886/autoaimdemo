#pragma once

#include <windows.h>
#include <avrt.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>

// 亲和工具：构建掩码 + 应用到当前线程（与进程亲和取交集）
static inline uint64_t BuildMaskFromCores(const std::vector<unsigned>& cores) {
    uint64_t m = 0;
    for (unsigned c : cores) if (c < 64) m |= (1ull << c);
    return m;
}
static inline bool ApplyThreadAffinityMask(uint64_t desiredMask, bool verbose=true) {
    if (!desiredMask) return false;
    DWORD_PTR procMask=0, sysMask=0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &procMask, &sysMask)) {
        if (verbose) std::cerr << "[AFF] GetProcessAffinityMask failed: " << GetLastError() << std::endl;
        return false;
    }
    DWORD_PTR finalMask = (DWORD_PTR)(desiredMask & (uint64_t)procMask);
    if (!finalMask) {
        if (verbose) std::cerr << "[AFF] empty after intersect with process mask" << std::endl;
        return false;
    }
    if (!SetThreadAffinityMask(GetCurrentThread(), finalMask)) {
        if (verbose) std::cerr << "[AFF] SetThreadAffinityMask failed: " << GetLastError() << std::endl;
        return false;
    }
    if (verbose) std::cout << "[AFF] Thread pinned. mask=0x" << std::hex << (uint64_t)finalMask << std::dec << std::endl;
    return true;
}

class Config {
public:
    // 基础日志控制
    bool verbose_init_log = true;                // 是否打印初始化阶段的详细日志
    int  capture_timeout_ms = 16;                // AcquireNextFrame 超时（毫秒）

    // ========== 0) 基础与可视化 ==========
    // 线程亲和与手动核心绑定（单组/SetThreadAffinityMask）
    bool enable_thread_affinity = false;            // 总开关：是否启用线程亲和
    uint64_t main_thread_affinity_mask = 0;         // 主线程掩码（bit 对应逻辑CPU）
    uint64_t overlay_thread_affinity_mask = 0;      // Overlay 线程掩码

    // 从环境变量覆盖（可选）：AIMAFF_MAIN, AIMAFF_OVERLAY
    void load_affinity_from_env() {
        if (const char* env = std::getenv("AIMAFF_MAIN")) {
            // 支持 0x 前缀十六进制，或逗号分隔数字列表
            parse_affinity_env(env, main_thread_affinity_mask);
        }
        if (const char* env = std::getenv("AIMAFF_OVERLAY")) {
            parse_affinity_env(env, overlay_thread_affinity_mask);
        }
    }
private:
    static void parse_affinity_env(const char* s, uint64_t& outMask) {
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
    const bool enable_visualization = true;
    const std::string window_name = "YOLO Real-time Detection";

    // ========== 1) 截屏与检测区域 ==========
    const int crop_size = 640;                    // 截取居中正方形区域尺寸
    const int max_lock_distance_pixels = 320;     // 目标距离中心阈值（像素）
    const float confidence_threshold = 0.6f;      // 检测阈值

    // ========== 2) 操作与热键 ==========
    const int smooth_aim_key = VK_RBUTTON;        // 按住平滑跟随
    const int single_shot_key = VK_F8;            // 单次移动测试

    // ========== 3) 瞄准与视角参数 ==========
    const double target_y_ratio = 0.6;            // 目标Y锚定比例（命中头/胸）
    const double pixels_for_360_turn = 3000;      // 鼠标360度所需像素（用于校正）
    const double horizontal_fov = 75.0;           // 游戏水平FOV（度）

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
    std::vector<std::string> class_names = { "enemy","teammate"};
    std::vector<int> target_class_ids = {0};

    // ========== 6) 运行时日志/统计 ==========
    // 6.1 延迟统计（粗粒度：capture/detect/mouse/loop）
    bool enable_latency_log = true;               // 打印延迟统计
    int  latency_log_interval_frames = 60;        // 每 N 帧打印一次
    bool latency_only_on_captured = true;         // 仅统计成功截屏的帧
    // 6.2 采集分步计时（细粒度：acquire/qi/copy/map/cvt/unmap/total）
    bool enable_capture_step_timing = true;       // 启用细粒度计时
    int  capture_timing_interval_frames = 60;     // [CAPTURE PERF] 打印周期（帧）
    bool capture_timing_every_frame = false;      // 每帧打印（调试用）
    bool capture_timing_only_on_success = true;   // 仅统计成功帧
    // 6.3 TensorRT-YOLO 内部性能报告
    bool enable_performance_report = true;
    int  perf_report_interval_frames = 60;        // TRTYOLO 报告频率（帧）
    bool perf_report_every_frame = false;

    // ========== 7) 优化启用报告（启动与周期性） ==========
    bool enable_optimization_report = true;       // 总开关：是否打印优化启用信息
    bool enable_capture_opt_report = true;        // 打印捕获链路优化信息
    bool enable_system_opt_report = true;         // 打印系统/调度优化信息
    bool optimization_report_verbose = true;      // 详细：打印OK/failed等细节
    int  optimization_report_interval_frames = 60;// [CAPTURE OPT] 打印周期（帧）

    // ========== 8) Overlay 动画（强制 DWM 产帧，诊断/兜底） ==========
    bool enable_overlay_animator = true;          // 启用极小透明 overlay，周期性更新以产生脏矩形
    // 更新间隔（单位：100ns；1ms=10,000×100ns）。推荐范围：5ms(50,000) ~ 16.7ms(167,000)。
    // 默认为 16ms（160,000），避免过高频率占用CPU。
    long long overlay_update_interval_100ns = 110; 
    // 自旋补偿预算（单位：微秒），>0 表示在等待定时器后进行短暂自旋以提升精度。推荐 0~200us。
    long long overlay_spin_budget_us = 2;          
    // 像素大小（单位：像素）。推荐 1 或 2。
    int  overlay_pixel_size = 2;                  
    // 屏幕坐标（单位：像素）。
    int  overlay_pos_x = 0;                       
    int  overlay_pos_y = 0;                       

    // ========== 9) 线程调度（MMCSS） ==========
    bool enable_mmcss_for_main_loop = true;       // 在主循环线程启用 MMCSS
    std::wstring mmcss_task_name = L"Pro Audio"; // 任务类别（"Games"/"Pro Audio"/"Capture"等）
    AVRT_PRIORITY mmcss_priority = AVRT_PRIORITY_HIGH; // MMCSS 线程优先级

    // ========== 9.5) 线程内存优先级 ==========
    bool enable_thread_memory_priority = true;    // 启用后将主线程 MemoryPriority 显式设为 NORMAL（保证不被降级）

    // ========== 10) DXGI 设备优先级/延迟 ==========
    // GPU 线程优先级（范围 -1..7；0=normal；>0 更高）。建议：0~3 常规，5~7 谨慎使用。
    int dxgi_gpu_thread_priority = 7;
    // 最大帧延迟（队列的最大帧数），推荐 1~3。1 通常最低延迟但可能影响吞吐。
    int dxgi_max_frame_latency = 1;

    // ========== 辅助 ==========
    bool is_target_class(int cid) const {
        return std::find(target_class_ids.begin(), target_class_ids.end(), cid) != target_class_ids.end();
    }
    std::string class_name_or_id(int cid) const {
        if (cid >= 0 && cid < (int)class_names.size() && !class_names[cid].empty()) return class_names[cid];
        return std::string("id=") + std::to_string(cid);
    }
};
