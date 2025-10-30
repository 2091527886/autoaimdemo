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

// Forward declare Config for use in OverlayAnim
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
#include "Reporting.hpp"
#include "ComHelpers.hpp"
#include "Detection.hpp"

// Forward Declarations
struct Timing;
class Config;
class ScreenCapturer;
class ObjDet;
class Mouse;
class AimAssistant;

// --- 辅助宏 ---
// --- 数据结构定义 ---



// --- PID 控制器 (moved to PidController.hpp/.cpp) ---


// --- 2. 鼠标控制器 (Mouse) ---
#include "OverlayAnimator.hpp"
#include "Mouse.hpp"
#include "HotkeyManager.hpp"
#include "ObjDet.hpp"
#include "ScreenCapturer.hpp"

// --- 3. 屏幕捕捉器 (ScreenCapturer) - 优化版本 ---
// (moved to ScreenCapturer.hpp/.cpp)

// --- 4. 目标检测器 (ObjDet) ---
/* moved to ObjDet.hpp/.cpp */

// --- 5. 核心协调器 (AimAssistant) ---
class AimAssistant {
public:
    AimAssistant()
        : cfg(),
        cap(cfg, cfg.crop_size_px, cfg.crop_size_px), // 传递 cfg + crop_size
        od(cfg),
        mouse(),
        keys(cfg),
        cropRect((cap.w() - cfg.crop_size_px) / 2, (cap.h() - cfg.crop_size_px) / 2, cfg.crop_size_px, cfg.crop_size_px),
        center(cfg.crop_size_px / 2, cfg.crop_size_px / 2),
        overlayOn(cfg.enable_visual),
        latAgg(static_cast<size_t>(std::max(1, cfg.log_latency_every_n)))
    {
        if (cap.w() < cfg.crop_size_px || cap.h() < cfg.crop_size_px) {
            throw std::runtime_error("Screen resolution is smaller than configured crop_size_px.");
        }
        mouseFactor = (cfg.hfov_deg / static_cast<double>(cap.w()))
            * (cfg.px_per_360 / 360.0);
        if (overlayOn) {
            cv::namedWindow(cfg.window_title, cv::WINDOW_AUTOSIZE);
        }
        std::cout << "--- Detect on centered " << cfg.crop_size_px << "x" << cfg.crop_size_px << " region. ---" << std::endl;
        showHelp();

        // 初始化 PID 控制器（像素域）。输出再乘以 mouse_correction_factor 转为鼠标单位。
        pidX.setK(cfg.pid_kp_x, cfg.pid_ki_x, cfg.pid_kd_x);
        pidX.setLim(cfg.pid_output_limit, cfg.pid_integral_limit);
        pidY.setK(cfg.pid_kp_y, cfg.pid_ki_y, cfg.pid_kd_y);
        pidY.setLim(cfg.pid_output_limit, cfg.pid_integral_limit);
    }

    ~AimAssistant() {
        if (overlayOn) cv::destroyAllWindows();
    }

    void run() {
        while (true) {
            auto loop_start_time = std::chrono::high_resolution_clock::now();

            FrameLat fl; // 本帧延迟统计

            // 1) 截屏计时
            auto t_cap_start = std::chrono::high_resolution_clock::now();
            bool got = cap.capture(frame, cropRect);
            auto t_cap_end = std::chrono::high_resolution_clock::now();
            fl.cap_ms = std::chrono::duration<double, std::milli>(t_cap_end - t_cap_start).count();
            if (!got) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                // 仅统计成功截屏帧：直接继续下一帧
                if (cfg.log_latency && !cfg.latency_on_capture_only) {
                    // 如果也要统计失败帧，则把仅有 capture_ms 的 fl 加入
                    latAgg.push(fl);
                }
                continue;
            }

            // 2) 检测计时（外层整体）
            auto t_det_start = std::chrono::high_resolution_clock::now();
            od.detect(frame, dets, cfg, tm);
            auto t_det_end = std::chrono::high_resolution_clock::now();
            fl.det_ms = std::chrono::duration<double, std::milli>(t_det_end - t_det_start).count();

            // 3) 鼠标控制计时（整体阶段）
            auto t_mouse_start = std::chrono::high_resolution_clock::now();
            Detection* target = findTarget();
            handleMouse(target);
            auto t_mouse_end = std::chrono::high_resolution_clock::now();
            fl.mou_ms = std::chrono::duration<double, std::milli>(t_mouse_end - t_mouse_start).count();

            auto loop_end_time = std::chrono::high_resolution_clock::now();
            tm.loop_ms = std::chrono::duration<double, std::milli>(loop_end_time - loop_start_time).count();
            fl.loop_ms = tm.loop_ms;

            // 计算 dt（秒）供 PID 使用
            if (cfg.pid_use_measured_dt) {
                dt = std::max(0.001, tm.loop_ms / 1000.0);
            } else {
                dt = std::max(0.001, cfg.pid_fixed_dt_ms / 1000.0);
            }
            updateStats();

            if (cfg.log_latency) {
                latAgg.push(fl);
            }

            if (overlayOn) {
                render(target);
            }
            char key = static_cast<char>(cv::waitKey(1));
            if (key == 27) { // ESC
                std::cout << "\nESC pressed. Exiting..." << std::endl;
                break;
            }
            if (key == 'v' || key == 'V') {
                toggleOverlay();
            }
        }
    }

private:
    void showHelp() {
        std::cout << "\n--- Controls ---" << std::endl;

    }

    Detection* findTarget() {
        Detection* target = nullptr;
        double min_dist_to_center = cfg.lock_radius_px;
        for (auto& det : dets) {
            if (!cfg.is_target(det.class_id)) continue; // 过滤非目标类别
            cv::Point box_center(det.box.x + det.box.width / 2, det.box.y + det.box.height / 2);
            double dist = std::hypot(box_center.x - center.x, box_center.y - center.y);
            if (dist < min_dist_to_center) {
                min_dist_to_center = dist;
                target = &det;
            }
        }
        return target;
    }

    void handleMouse(const Detection* target) {
        // 若无目标或未按住按键：释放 PID 积分，避免历史积分殘留
        bool smooth_pressed = keys.SmoothPressed();
        if (!target || !smooth_pressed) {
            pidX.reset();
            pidY.reset();
            // 仍保留单次 F8 逻辑
            if (!target) return;
        }

        if (smooth_pressed) {
            cv::Point target_point(
                target->box.x + target->box.width / 2,
                target->box.y + static_cast<int>(target->box.height * cfg.y_anchor_ratio)
            );
            double err_x = static_cast<double>(target_point.x - center.x);
            double err_y = static_cast<double>(target_point.y - center.y);

            // PID 在像素域输出，再乘鼠标校正因子转为鼠标单位
            double out_x_pix = pidX.step(err_x, dt);
            double out_y_pix = pidY.step(err_y, dt);
            int move_x = static_cast<int>(out_x_pix * mouseFactor);
            int move_y = static_cast<int>(out_y_pix * mouseFactor);

            if (move_x != 0 || move_y != 0) {
                mouse.Move(move_x, move_y);
            }
        }

        // 单发模式：保持边沿触发语义
        if (keys.SingleShotJustPressed()) {
            cv::Point target_point(
                target->box.x + target->box.width / 2,
                target->box.y + static_cast<int>(target->box.height * cfg.y_anchor_ratio)
            );
            double dx_pixels = static_cast<double>(target_point.x - center.x);
            double dy_pixels = static_cast<double>(target_point.y - center.y);
            double corrected_dx = dx_pixels * mouseFactor;
            double corrected_dy = dy_pixels * mouseFactor;
            int move_x = static_cast<int>(corrected_dx);
            int move_y = static_cast<int>(corrected_dy);
            if (move_x != 0 || move_y != 0) {
                mouse.Move(move_x, move_y);
            }
        }
    }

    void updateStats() {
        const double smoothing_factor = 0.05;
        if (firstFrame) {
            msTotal = tm.loop_ms;
            msInfer = tm.infer_ms;
            msPost = tm.post_ms;
            firstFrame = false;
        }
        else {
            msTotal = smoothing_factor * tm.loop_ms + (1.0 - smoothing_factor) * msTotal;
            msInfer = smoothing_factor * tm.infer_ms + (1.0 - smoothing_factor) * msInfer;
            msPost = smoothing_factor * tm.post_ms + (1.0 - smoothing_factor) * msPost;
        }

    }

    void render(const Detection* best_target) {
        cv::circle(frame, center, static_cast<int>(cfg.lock_radius_px), cv::Scalar(255, 255, 0), 1);
        for (const auto& det : dets) {
            // 可视化：所有检测都显示；追踪类加星号，当前目标红框
            cv::Scalar color = (&det == best_target) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            cv::rectangle(frame, det.box, color, 2);
            std::string label = cfg.name_or_id(det.class_id) + (cfg.is_target(det.class_id) ? "*" : "") +
                                 " " + cv::format("%.2f", det.confidence);
            cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
        if (msTotal > 0) {
            double smoothed_fps = 1000.0 / msTotal;
            std::ostringstream stats_stream;
            stats_stream << std::fixed << std::setprecision(1) << "FPS: " << smoothed_fps;
            cv::putText(frame, stats_stream.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
        }
        cv::imshow(cfg.window_title, frame);
    }

    void toggleOverlay() {
        overlayOn = !overlayOn;
        if (overlayOn) {
            cv::namedWindow(cfg.window_title, cv::WINDOW_AUTOSIZE);
            std::cout << "\n[INFO] Visualization toggled ON." << std::endl;
        }
        else {
            cv::destroyAllWindows();
            std::cout << "\n[INFO] Visualization toggled OFF. Console stats will continue." << std::endl;
        }
    }

    Config cfg;
    ScreenCapturer cap;
    ObjDet od;
    Mouse mouse;
    HotkeyManager keys;
    const cv::Rect cropRect;
    const cv::Point center;
    double mouseFactor = 1.0;
    cv::Mat frame;
    std::vector<Detection> dets;
    Timing tm;
    bool overlayOn;
    double msTotal = 0.0, msInfer = 0.0, msPost = 0.0;
    bool firstFrame = true;
    // 延迟统计
    LatAgg latAgg;

    // PID 控制
    PidController pidX;
    PidController pidY;
    double dt = 1.0/60.0;
};

// --- 主函数 ---
int main() {
    Config boot_cfg; // KEEP
    // 从环境变量读取可选亲和配置
    boot_cfg.load_affinity_env();
    if (boot_cfg.enable_affinity && boot_cfg.affinity_main_mask) {
        ApplyThreadAffinityMask(boot_cfg.affinity_main_mask,
            boot_cfg.opt_report && boot_cfg.opt_report_system);
    }

    OverlayAnim overlay;
    // Disable process/thread power throttling and background limits as early as possible
    DisablePowerThrottlingForProcessAndThread();
    // Raise process and main thread priority early
    SetProcPrioHigh();
    SetThreadPrioHighest();

    // Enable MMCSS for main loop thread based on default Config values
   
    HANDLE hMmcss = EnableMmcssIfConfigured(boot_cfg.mmcss_main,
                                            boot_cfg.mmcss_task.c_str(),
                                            boot_cfg.mmcss_prio);

   // 提升主线程内存优先级 & 禁用进程级 PriorityBoost
   if (boot_cfg.enable_thread_mem_prio) {
       SetThreadMemPrioNormal();
       SetProcPrioBoostDisabled();
   }

    // Start tiny overlay animator (forces DWM updates) if enabled
    if (boot_cfg.overlay_anim) {
        overlay.start(boot_cfg);
        metrics::SystemBootReporting::PrintOverlayBanner(boot_cfg, true);
    } else {
        metrics::SystemBootReporting::PrintOverlayBanner(boot_cfg, false);
    }

    metrics::SystemBootReporting::PrintSchedulingBanner(boot_cfg, hMmcss);

    try {
        AimAssistant assistant;
        assistant.run();
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
