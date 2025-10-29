#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <memory>
#include <sstream> // For std::ostringstream
#include <thread>  // For std::this_thread
#include <limits>
#include <tuple>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
// Windows & DirectX
#include <windows.h>
#include <winnt.h> // for MEMORY_PRIORITY_INFORMATION and MEMORY_PRIORITY_* constants
#include <d3d11.h>
#include <dxgi1_2.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#include <avrt.h>
#pragma comment(lib, "Avrt.lib")

// --- Priority helpers moved to separate modules ---
/* moved */


// 将当前线程内存优先级显式设置为 NORMAL（保证不被降级）
/* moved */
// 禁用进程 PriorityBoost，避免在系统内存压力下被降低优先级
/* moved */

// Forward declare Config for use in OverlayAnimator
class Config;
/* moved: MMCSS utilities now in MmcssUtil */
/* moved */
void DisableMmcss(HANDLE h);
/* moved */

// TensorRT-YOLO
#include <trtyolo.hpp>
#pragma comment(lib, "trtyolo.lib")

// OpenCV
#include <opencv2/opencv.hpp>
#ifdef _DEBUG
#pragma comment(lib, "opencv_world4110d.lib") // 请根据你的OpenCV版本修改
#else
#pragma comment(lib, "opencv_world4110.lib") // 请根据你的OpenCV版本修改
#endif

// Input Simulator (确保头文件在项目路径中)
#include "InputSimulator.hpp"
#include "Config.hpp"
#include "Timing.hpp"
#include "PidController.hpp"
#include "SystemPriority.hpp"
#include "PowerThrottling.hpp"
#include "MmcssUtil.hpp"
#include "DxgiTuning.hpp"
#include "ThreadAffinity.hpp"
#include "metrics/CaptureReporting.hpp"
#include "metrics/TrtPerfReporting.hpp"
#include "metrics/SystemBootReporting.hpp"
#include "ComHelpers.hpp"
#include "Detection.hpp"

// Forward Declarations
struct TimingDetails;
class Config;
class ScreenCapturer;
class ObjectDetector;
class MouseController;
class AimAssistant;

// --- 辅助宏 ---
// --- 数据结构定义 ---



// --- PID 控制器 (moved to PidController.hpp/.cpp) ---


// --- 2. 鼠标控制器 (MouseController) ---
#include "OverlayAnimator.hpp"
#include "MouseController.hpp"
#include "HotkeyManager.hpp"
#include "ObjectDetector.hpp"
#include "ScreenCapturer.hpp"

// --- 3. 屏幕捕捉器 (ScreenCapturer) - 优化版本 ---
// (moved to ScreenCapturer.hpp/.cpp)

// --- 4. 目标检测器 (ObjectDetector) ---
/* moved to ObjectDetector.hpp/.cpp */

// --- 5. 核心协调器 (AimAssistant) ---
class AimAssistant {
public:
    AimAssistant()
        : cfg(),
        capturer(cfg, cfg.crop_size, cfg.crop_size), // 传递 cfg + crop_size
        detector(cfg),
        mouse(),
        hotkeys(cfg),
        crop_region((capturer.getWidth() - cfg.crop_size) / 2, (capturer.getHeight() - cfg.crop_size) / 2, cfg.crop_size, cfg.crop_size),
        crop_center(cfg.crop_size / 2, cfg.crop_size / 2),
        is_visualizing(cfg.enable_visualization),
        aggregator_(static_cast<size_t>(std::max(1, cfg.latency_log_interval_frames)))
    {
        if (capturer.getWidth() < cfg.crop_size || capturer.getHeight() < cfg.crop_size) {
            throw std::runtime_error("Screen resolution is smaller than configured crop_size.");
        }
        mouse_correction_factor = (cfg.horizontal_fov / static_cast<double>(capturer.getWidth()))
            * (cfg.pixels_for_360_turn / 360.0);
        if (is_visualizing) {
            cv::namedWindow(cfg.window_name, cv::WINDOW_AUTOSIZE);
        }
        std::cout << "--- Detection will run on a centered " << cfg.crop_size << "x" << cfg.crop_size << " region. ---" << std::endl;
        print_instructions();

        // 初始化 PID 控制器（像素域）。输出再乘以 mouse_correction_factor 转为鼠标单位。
        pid_x.SetGains(cfg.pid_kp_x, cfg.pid_ki_x, cfg.pid_kd_x);
        pid_x.SetLimits(cfg.pid_output_limit, cfg.pid_integral_limit);
        pid_y.SetGains(cfg.pid_kp_y, cfg.pid_ki_y, cfg.pid_kd_y);
        pid_y.SetLimits(cfg.pid_output_limit, cfg.pid_integral_limit);
    }

    ~AimAssistant() {
        if (is_visualizing) cv::destroyAllWindows();
    }

    void Run() {
        while (true) {
            auto loop_start_time = std::chrono::high_resolution_clock::now();

            FrameLatency fl; // 本帧延迟统计

            // 1) 截屏计时
            auto t_cap_start = std::chrono::high_resolution_clock::now();
            bool got = capturer.CaptureFrame(captured_frame, crop_region);
            auto t_cap_end = std::chrono::high_resolution_clock::now();
            fl.capture_ms = std::chrono::duration<double, std::milli>(t_cap_end - t_cap_start).count();
            if (!got) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                // 仅统计成功截屏帧：直接继续下一帧
                if (cfg.enable_latency_log && !cfg.latency_only_on_captured) {
                    // 如果也要统计失败帧，则把仅有 capture_ms 的 fl 加入
                    aggregator_.add(fl);
                }
                continue;
            }

            // 2) 检测计时（外层整体）
            auto t_det_start = std::chrono::high_resolution_clock::now();
            detector.Detect(captured_frame, detections, cfg, timings);
            auto t_det_end = std::chrono::high_resolution_clock::now();
            fl.detect_ms = std::chrono::duration<double, std::milli>(t_det_end - t_det_start).count();

            // 3) 鼠标控制计时（整体阶段）
            auto t_mouse_start = std::chrono::high_resolution_clock::now();
            Detection* best_target = findBestTarget();
            handleMouseInput(best_target);
            auto t_mouse_end = std::chrono::high_resolution_clock::now();
            fl.mouse_ms = std::chrono::duration<double, std::milli>(t_mouse_end - t_mouse_start).count();

            auto loop_end_time = std::chrono::high_resolution_clock::now();
            timings.total_loop_ms = std::chrono::duration<double, std::milli>(loop_end_time - loop_start_time).count();
            fl.loop_ms = timings.total_loop_ms;

            // 计算 dt（秒）供 PID 使用
            if (cfg.pid_use_measured_dt) {
                pid_dt_s = std::max(0.001, timings.total_loop_ms / 1000.0);
            } else {
                pid_dt_s = std::max(0.001, cfg.pid_fixed_dt_ms / 1000.0);
            }
            updateAndPrintStats();

            if (cfg.enable_latency_log) {
                aggregator_.add(fl);
            }

            if (is_visualizing) {
                handleVisualization(best_target);
            }
            char key = static_cast<char>(cv::waitKey(1));
            if (key == 27) { // ESC
                std::cout << "\nESC pressed. Exiting..." << std::endl;
                break;
            }
            if (key == 'v' || key == 'V') {
                toggleVisualization();
            }
        }
    }

private:
    void print_instructions() {
        std::cout << "\n--- Controls ---" << std::endl;

    }

    Detection* findBestTarget() {
        Detection* target = nullptr;
        double min_dist_to_center = cfg.max_lock_distance_pixels;
        for (auto& det : detections) {
            if (!cfg.is_target_class(det.class_id)) continue; // 过滤非目标类别
            cv::Point box_center(det.box.x + det.box.width / 2, det.box.y + det.box.height / 2);
            double dist = std::hypot(box_center.x - crop_center.x, box_center.y - crop_center.y);
            if (dist < min_dist_to_center) {
                min_dist_to_center = dist;
                target = &det;
            }
        }
        return target;
    }

    void handleMouseInput(const Detection* target) {
        // 若无目标或未按住按键：释放 PID 积分，避免历史积分殘留
        bool smooth_pressed = hotkeys.SmoothPressed();
        if (!target || !smooth_pressed) {
            pid_x.Reset();
            pid_y.Reset();
            // 仍保留单次 F8 逻辑
            if (!target) return;
        }

        if (smooth_pressed) {
            cv::Point target_point(
                target->box.x + target->box.width / 2,
                target->box.y + static_cast<int>(target->box.height * cfg.target_y_ratio)
            );
            double err_x = static_cast<double>(target_point.x - crop_center.x);
            double err_y = static_cast<double>(target_point.y - crop_center.y);

            // PID 在像素域输出，再乘鼠标校正因子转为鼠标单位
            double out_x_pix = pid_x.Update(err_x, pid_dt_s);
            double out_y_pix = pid_y.Update(err_y, pid_dt_s);
            int move_x = static_cast<int>(out_x_pix * mouse_correction_factor);
            int move_y = static_cast<int>(out_y_pix * mouse_correction_factor);

            if (move_x != 0 || move_y != 0) {
                mouse.MoveRelative(move_x, move_y);
            }
        }

        // 单发模式：保持边沿触发语义
        if (hotkeys.SingleShotJustPressed()) {
            cv::Point target_point(
                target->box.x + target->box.width / 2,
                target->box.y + static_cast<int>(target->box.height * cfg.target_y_ratio)
            );
            double dx_pixels = static_cast<double>(target_point.x - crop_center.x);
            double dy_pixels = static_cast<double>(target_point.y - crop_center.y);
            double corrected_dx = dx_pixels * mouse_correction_factor;
            double corrected_dy = dy_pixels * mouse_correction_factor;
            int move_x = static_cast<int>(corrected_dx);
            int move_y = static_cast<int>(corrected_dy);
            if (move_x != 0 || move_y != 0) {
                mouse.MoveRelative(move_x, move_y);
            }
        }
    }

    void updateAndPrintStats() {
        const double smoothing_factor = 0.05;
        if (is_first_frame) {
            smoothed_total = timings.total_loop_ms;
            smoothed_inf = timings.inference_ms;
            smoothed_post = timings.postprocess_ms;
            is_first_frame = false;
        }
        else {
            smoothed_total = smoothing_factor * timings.total_loop_ms + (1.0 - smoothing_factor) * smoothed_total;
            smoothed_inf = smoothing_factor * timings.inference_ms + (1.0 - smoothing_factor) * smoothed_inf;
            smoothed_post = smoothing_factor * timings.postprocess_ms + (1.0 - smoothing_factor) * smoothed_post;
        }

    }

    void handleVisualization(const Detection* best_target) {
        cv::circle(captured_frame, crop_center, static_cast<int>(cfg.max_lock_distance_pixels), cv::Scalar(255, 255, 0), 1);
        for (const auto& det : detections) {
            // 可视化：所有检测都显示；追踪类加星号，当前目标红框
            cv::Scalar color = (&det == best_target) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            cv::rectangle(captured_frame, det.box, color, 2);
            std::string label = cfg.class_name_or_id(det.class_id) + (cfg.is_target_class(det.class_id) ? "*" : "") +
                                 " " + cv::format("%.2f", det.confidence);
            cv::putText(captured_frame, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
        if (smoothed_total > 0) {
            double smoothed_fps = 1000.0 / smoothed_total;
            std::ostringstream stats_stream;
            stats_stream << std::fixed << std::setprecision(1) << "FPS: " << smoothed_fps;
            cv::putText(captured_frame, stats_stream.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
        }
        cv::imshow(cfg.window_name, captured_frame);
    }

    void toggleVisualization() {
        is_visualizing = !is_visualizing;
        if (is_visualizing) {
            cv::namedWindow(cfg.window_name, cv::WINDOW_AUTOSIZE);
            std::cout << "\n[INFO] Visualization toggled ON." << std::endl;
        }
        else {
            cv::destroyAllWindows();
            std::cout << "\n[INFO] Visualization toggled OFF. Console stats will continue." << std::endl;
        }
    }

    Config cfg;
    ScreenCapturer capturer;
    ObjectDetector detector;
    MouseController mouse;
    HotkeyManager hotkeys;
    const cv::Rect crop_region;
    const cv::Point crop_center;
    double mouse_correction_factor = 1.0;
    cv::Mat captured_frame;
    std::vector<Detection> detections;
    TimingDetails timings;
    bool is_visualizing;
    double smoothed_total = 0.0, smoothed_inf = 0.0, smoothed_post = 0.0;
    bool is_first_frame = true;
    // 延迟统计
    LatencyAggregator aggregator_;

    // PID 控制
    PidController pid_x;
    PidController pid_y;
    double pid_dt_s = 1.0/60.0;
};

// --- 主函数 ---
int main() {
    Config boot_cfg; // KEEP
    // 从环境变量读取可选亲和配置
    boot_cfg.load_affinity_from_env();
    if (boot_cfg.enable_thread_affinity && boot_cfg.main_thread_affinity_mask) {
        ApplyThreadAffinityMask(boot_cfg.main_thread_affinity_mask,
            boot_cfg.enable_optimization_report && boot_cfg.enable_system_opt_report);
    }

    OverlayAnimator overlay;
    // Disable process/thread power throttling and background limits as early as possible
    DisablePowerThrottlingForProcessAndThread();
    // Raise process and main thread priority early
    RaiseProcessPrioritySafely();
    RaiseThisThreadPrioritySafely();

    // Enable MMCSS for main loop thread based on default Config values
   
    HANDLE hMmcss = EnableMmcssIfConfigured(boot_cfg.enable_mmcss_for_main_loop,
                                            boot_cfg.mmcss_task_name.c_str(),
                                            boot_cfg.mmcss_priority);

   // 提升主线程内存优先级 & 禁用进程级 PriorityBoost
   if (boot_cfg.enable_thread_memory_priority) {
       SetThisThreadMemoryPriorityNormal();
       DisableProcessPriorityBoost();
   }

    // Start tiny overlay animator (forces DWM updates) if enabled
    if (boot_cfg.enable_overlay_animator) {
        overlay.start(boot_cfg);
        metrics::SystemBootReporting::PrintOverlayBanner(boot_cfg, true);
    } else {
        metrics::SystemBootReporting::PrintOverlayBanner(boot_cfg, false);
    }

    metrics::SystemBootReporting::PrintSchedulingBanner(boot_cfg, hMmcss);

    try {
        AimAssistant assistant;
        assistant.Run();
    }
    catch (const std::exception& e) {
        DisableMmcss(hMmcss);
        std::cerr << "\n\n[FATAL ERROR] An unrecoverable error occurred: " << e.what() << std::endl;
        std::cerr << "Please check the following:\n"
            << "Press Enter to exit." << std::endl;
        std::cin.get();
        return -1;
    }
    std::cout << "\nProgram finished successfully." << std::endl;
    overlay.stop();
    DisableMmcss(hMmcss);
    return 0;
}
