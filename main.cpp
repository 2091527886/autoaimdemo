#include <iostream>
#include "Log.hpp"
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <thread>
// OpenCV
#include <opencv2/opencv.hpp>
// Input Simulator (确保头文件在项目路径中)
#include "Config.hpp"
#include "Timing.hpp"
#include "PidController.hpp"
#include "SystemTuner.hpp"
#include "Reporting.hpp"
#include "Detection.hpp"

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

// --- 5. 核心协调器 (Aim) ---
class Aim {
public:
    Aim()
        : c(),
        cap(c, c.crop_size_px, c.crop_size_px), // 传递 cfg + crop_size
        od(c),
        m(),
        hk(c),
        crop((cap.w() - c.crop_size_px) / 2, (cap.h() - c.crop_size_px) / 2, c.crop_size_px, c.crop_size_px),
        ctr(c.crop_size_px / 2, c.crop_size_px / 2),
        vis(c.enable_visual),
        lat(static_cast<size_t>(std::max(1, c.log_latency_every_n)))
    {
        if (cap.w() < c.crop_size_px || cap.h() < c.crop_size_px) {
            throw std::runtime_error("Screen resolution is smaller than configured crop_size_px.");
        }
        mfac = (c.hfov_deg / static_cast<double>(cap.w()))
            * (c.px_per_360 / 360.0);
        if (vis) {
            cv::namedWindow(c.window_title, cv::WINDOW_AUTOSIZE);
        }
        lg::o() << "Detect center " << c.crop_size_px << "x" << c.crop_size_px << lg::endl;
        // 初始化 PID 控制器（像素域）。输出再乘以 mouse_correction_factor 转为鼠标单位。
        px.setK(c.pid_kp_x, c.pid_ki_x, c.pid_kd_x);
        px.setLim(c.pid_output_limit, c.pid_integral_limit);
        py.setK(c.pid_kp_y, c.pid_ki_y, c.pid_kd_y);
        py.setLim(c.pid_output_limit, c.pid_integral_limit);
    }

    ~Aim() {
        if (vis) cv::destroyAllWindows();
    }

    void loop() {
        while (true) {
            auto loop_start_time = std::chrono::high_resolution_clock::now();

            FrameLat fl; // 本帧延迟统计

            // 1) 截屏计时
            auto t_cap_start = std::chrono::high_resolution_clock::now();
            bool got = cap.capture(img, crop);
            auto t_cap_end = std::chrono::high_resolution_clock::now();
            fl.cap_ms = std::chrono::duration<double, std::milli>(t_cap_end - t_cap_start).count();
            if (!got) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                // 仅统计成功截屏帧：直接继续下一帧
                if (c.log_latency && !c.latency_on_capture_only) {
                    // 如果也要统计失败帧，则把仅有 capture_ms 的 fl 加入
                    lat.push(fl);
                }
                continue;
            }

            // 2) 检测计时（外层整体）
            auto t_det_start = std::chrono::high_resolution_clock::now();
            od.detect(img, ds, c, tm);
            auto t_det_end = std::chrono::high_resolution_clock::now();
            fl.det_ms = std::chrono::duration<double, std::milli>(t_det_end - t_det_start).count();

            // 3) 鼠标控制计时（整体阶段）
            auto t_mouse_start = std::chrono::high_resolution_clock::now();
            Detection* target = pick();
            move(target);
            auto t_mouse_end = std::chrono::high_resolution_clock::now();
            fl.mou_ms = std::chrono::duration<double, std::milli>(t_mouse_end - t_mouse_start).count();

            auto loop_end_time = std::chrono::high_resolution_clock::now();
            tm.loop_ms = std::chrono::duration<double, std::milli>(loop_end_time - loop_start_time).count();
            fl.loop_ms = tm.loop_ms;

            // 计算 dt（秒）供 PID 使用
            if (c.pid_use_measured_dt) {
                dt = std::max(0.001, tm.loop_ms / 1000.0);
            } else {
                dt = std::max(0.001, c.pid_fixed_dt_ms / 1000.0);
            }
            stats();

            if (c.log_latency) {
                lat.push(fl);
            }

            if (vis) {
                draw(target);
            }
            char key = static_cast<char>(cv::waitKey(1));
            if (key == 27) { // ESC
                lg::o() << lg::nl << P_I "ESC, exit" << lg::endl;
                break;
            }
            if (key == 'v' || key == 'V') {
                toggle();
            }
        }
    }

private:
    Detection* pick() {
        Detection* target = nullptr;
        double min_dist_to_center = c.lock_radius_px;
        for (auto& det : ds) {
            if (!c.is_target(det.class_id)) continue; // 过滤非目标类别
            cv::Point box_center(det.box.x + det.box.width / 2, det.box.y + det.box.height / 2);
            double dist = std::hypot(box_center.x - ctr.x, box_center.y - ctr.y);
            if (dist < min_dist_to_center) {
                min_dist_to_center = dist;
                target = &det;
            }
        }
        return target;
    }

    void move(const Detection* target) {
        // 若无目标或未按住按键：释放 PID 积分，避免历史积分殘留
        bool smooth_pressed = hk.SmoothPressed();
        if (!target || !smooth_pressed) {
            px.reset();
            py.reset();
            // 仍保留单次 F8 逻辑
            if (!target) return;
        }

        if (smooth_pressed) {
            cv::Point target_point(
                target->box.x + target->box.width / 2,
                target->box.y + static_cast<int>(target->box.height * c.y_anchor_ratio)
            );
            double err_x = static_cast<double>(target_point.x - ctr.x);
            double err_y = static_cast<double>(target_point.y - ctr.y);

            // PID 在像素域输出，再乘鼠标校正因子转为鼠标单位
            double out_x_pix = px.step(err_x, dt);
            double out_y_pix = py.step(err_y, dt);
            int move_x = static_cast<int>(out_x_pix * mfac);
            int move_y = static_cast<int>(out_y_pix * mfac);

            if (move_x != 0 || move_y != 0) {
                m.Move(move_x, move_y);
            }
        }

        // 单发模式：保持边沿触发语义
        if (hk.SingleShotJustPressed()) {
            cv::Point target_point(
                target->box.x + target->box.width / 2,
                target->box.y + static_cast<int>(target->box.height * c.y_anchor_ratio)
            );
            double dx_pixels = static_cast<double>(target_point.x - ctr.x);
            double dy_pixels = static_cast<double>(target_point.y - ctr.y);
            double corrected_dx = dx_pixels * mfac;
            double corrected_dy = dy_pixels * mfac;
            int move_x = static_cast<int>(corrected_dx);
            int move_y = static_cast<int>(corrected_dy);
            if (move_x != 0 || move_y != 0) {
                m.Move(move_x, move_y);
            }
        }
    }

    void stats() {
        const double smoothing_factor = 0.05;
        if (first) {
            tTot = tm.loop_ms;
            first = false;
        }
        else {
            tTot = smoothing_factor * tm.loop_ms + (1.0 - smoothing_factor) * tTot;
        }

    }

    void draw(const Detection* tgtBest) {
        cv::circle(img, ctr, static_cast<int>(c.lock_radius_px), cv::Scalar(255, 255, 0), 1);
        for (const auto& det : ds) {
            // 可视化：所有检测都显示；追踪类加星号，当前目标红框
            cv::Scalar color = (&det == tgtBest) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            cv::rectangle(img, det.box, color, 2);
            std::string label = c.name_or_id(det.class_id) + (c.is_target(det.class_id) ? "*" : "") +
                                 " " + cv::format("%.2f", det.confidence);
            cv::putText(img, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
        if (tTot > 0) {
            double smoothed_fps = 1000.0 / tTot;
            std::ostringstream stats_stream;
            stats_stream << lg::fx1 << "FPS: " << smoothed_fps << lg::df;
            cv::putText(img, stats_stream.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
        }
        cv::imshow(c.window_title, img);
    }

    void toggle() {
        vis = !vis;
        if (vis) {
            cv::namedWindow(c.window_title, cv::WINDOW_AUTOSIZE);
            lg::o() << lg::nl << P_I "Vis ON" << lg::endl;
        }
        else {
            cv::destroyAllWindows();
            lg::o() << lg::nl << P_I "Vis OFF" << lg::endl;
        }
    }

    Config c;
    ScreenCapturer cap;
    ObjDet od;
    Mouse m;
    HotkeyManager hk;
    const cv::Rect crop;
    const cv::Point ctr;
    double mfac = 1.0;
    cv::Mat img;
    std::vector<Detection> ds;
    Timing tm;
    bool vis;
    double tTot = 0.0;
    bool first = true;
    // 延迟统计
    LatAgg lat;

    // PID 控制
    PidController px;
    PidController py;
    double dt = 1.0/60.0;
};

// --- 主函数 ---
int main() {
    Config boot_cfg; // KEEP
    // 从环境变量读取可选亲和配置
    boot_cfg.load_affinity_env();
    if (boot_cfg.enable_affinity && boot_cfg.affinity_main_mask) {
        setAffinity(boot_cfg.affinity_main_mask,
            boot_cfg.opt_report && boot_cfg.opt_report_system);
    }

    OverlayAnim overlay;
    // Disable process/thread power throttling and background limits as early as possible
    disablePowerThrottle();
    // Raise process and main thread priority early
    SetProcPrioHigh();
    SetThreadPrioHighest();

    // Enable MMCSS for main loop thread based on default Config values
   
    HANDLE hMmcss = MmcssEnable(boot_cfg.mmcss_main,
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
        Aim assistant;
        assistant.loop();
    }
    catch (const std::exception& e) {
        MmcssDisable(hMmcss);
        lg::ml(P_F, std::string("Fatal: ") + e.what(), "Check issues.", "Press Enter to exit.");
        std::cin.get();
        return -1;
    }
    lg::o() << lg::nl << P_I "Done" << lg::endl;
    overlay.stop();
    MmcssDisable(hMmcss);
    return 0;
}
