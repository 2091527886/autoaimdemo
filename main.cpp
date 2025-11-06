#include <iostream>
#include "Log.hpp"
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <thread>
#include <atomic>
#include <windows.h>

extern std::atomic<bool> g_exit_req;
extern std::atomic<bool> g_exit_done;

#include <opencv2/opencv.hpp>
#include "Config.hpp"
#include "Timing.hpp"
#include "SystemTuner.hpp"
#include "Reporting.hpp"
#include "Detection.hpp"






#include "Mouse.hpp"
#include "AimAlgo.hpp"
#include "HotkeyManager.hpp"
#include "ScreenCapturer.hpp"
#include "ObjDet.hpp"
#include "OvlAsync.hpp"
#include "LatCsvAsync.hpp"



// helpers
static Detection* pick_target(std::vector<Detection>& ds, const Config& c, const cv::Point& ctr) {
    Detection* target = nullptr;
    double min_dist_to_center = std::numeric_limits<double>::infinity();
    for (auto& det : ds) {
        if (!c.is_target(det.class_id)) continue;
        cv::Point box_center(det.box.x + det.box.width / 2, det.box.y + det.box.height / 2);
        double dist = std::hypot(box_center.x - ctr.x, box_center.y - ctr.y);
        if (dist < min_dist_to_center) { min_dist_to_center = dist; target = &det; }
    }
    return target;
}

class Aim {
public:
    // FPS 锁定/补偿控制（短名）
    struct FpsCtl {
        double tgt_ms = 8.0;
        long long spin_us = 0;
        size_t drop = 0;
        double ovr = 0.0;
        HANDLE ht = nullptr;
        explicit FpsCtl(const Config& cfg) {
            tgt_ms = 1000.0 / std::max(1, cfg.fps);
            spin_us = cfg.fps_spin_us;
            ht = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
            if (!ht) ht = CreateWaitableTimerW(NULL, FALSE, NULL);
        }
        ~FpsCtl() { if (ht) CloseHandle(ht); }
        double end(double proc_ms) {
            if (proc_ms <= tgt_ms) {
                double rem = tgt_ms - proc_ms;
                double sl = rem - (double)spin_us / 1000.0;
                if (sl > 0.0) {
                    if (ht) {
                        LARGE_INTEGER due; long long q100ns = (long long)(sl * 10000.0);
                        if (q100ns < 1) q100ns = 1; due.QuadPart = -q100ns;
                        if (SetWaitableTimer(ht, &due, 0, nullptr, nullptr, FALSE)) {
                            WaitForSingleObject(ht, INFINITE);
                        } else {
                            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(sl));
                        }
                    } else {
                        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(sl));
                    }
                }
                if (spin_us > 0) {
                    auto se = std::chrono::high_resolution_clock::now() + std::chrono::microseconds((long long)spin_us);
                    while (std::chrono::high_resolution_clock::now() < se) { YieldProcessor(); }
                }
                ovr = 0.0;
                return tgt_ms;
            } else {
                ovr = proc_ms - tgt_ms;
                drop++;
                return proc_ms;
            }
        }
    };

    Aim()
        : c(),
        cap(c, c.crop_size_px, c.crop_size_px),
        od(c),
        hk(c),
        crop((cap.w() - c.crop_size_px) / 2, (cap.h() - c.crop_size_px) / 2, c.crop_size_px, c.crop_size_px),
        ctr(c.crop_size_px / 2, c.crop_size_px / 2),
        vis(c.enable_visual),
        lat(static_cast<size_t>(std::max(1, c.log_latency_every_n))),
        fc(c),
        ovl(c),
        latcsv(c)
    {
        if (c.timing && c.lat_csv) latcsv.start(c.lat_csv_path);

        if (cap.w() < c.crop_size_px || cap.h() < c.crop_size_px) {
            throw std::runtime_error("Screen resolution is smaller than configured crop_size_px.");
        }
        mfac = (c.hfov_deg / static_cast<double>(cap.w()))
            * (c.px_per_360 / 360.0);
        if (vis) {
            cv::namedWindow(c.window_title, cv::WINDOW_AUTOSIZE);
        }
        lg::o() << "Detect center " << c.crop_size_px << "x" << c.crop_size_px << lg::endl;
        // no async ROI/start; capture on main thread
        /* inference on main thread: no det.start() */
        
        // init AimAlgo (PropAlgo-based)
        algo.setConfig(c);
        if (c.ovl) { lg::o() << P_B "Overlay=ON" << lg::endl; ovl.start(); }
        else { lg::o() << P_B "Overlay=OFF" << lg::endl; }
    }

    ~Aim() {
        if (c.timing && c.lat_csv) latcsv.stop();
        /* inference on main thread: no det.stop() */
        
        if (c.ovl) ovl.stop();
        if (vis) cv::destroyAllWindows();
    }

    void loop() {
        while (true) {
            if (g_exit_req.load(std::memory_order_relaxed)) { lg::o() << lg::nl << P_I "Signal exit" << lg::endl; break; }
            // no event pacing; capture on main thread
            auto loop_start_time = std::chrono::high_resolution_clock::now();

            FrameLat fl;

            // capture on main thread
            auto tcap0 = (c.timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{});
            bool got = cap.capture(img, crop);
            auto tcap1 = (c.timing ? std::chrono::high_resolution_clock::now() : tcap0);
            double cap_ms = c.timing ? std::chrono::duration<double, std::milli>(tcap1 - tcap0).count() : 0.0;
            if (!got) {
                fl.cap_ms = cap_ms;
                if (c.timing && c.log_latency && !c.latency_on_capture_only) {
                    lat.push(fl);
                }
                if (g_exit_req.load(std::memory_order_relaxed)) { lg::o() << lg::nl << P_I "Signal exit" << lg::endl; break; }
                continue;
            }
            c.frame_idx++; fl.idx = c.frame_idx;
            fl.cap_ms = cap_ms;

            // inference on main thread
            od.detect(img, ds, c, tm);
            det_ms_last = c.timing ? (tm.infer_ms + tm.post_ms) : 0.0;
            if (c.ovl) ovl.put(ds, crop);
            fl.det_ms = det_ms_last;

            
            
            bool smooth_pressed = hk.SmoothPressed();
            bool single = hk.SingleShotJustPressed();

            // 延后计算 dt：仅在即将执行算法时按需计算
            auto calc_dt = [&](){
                if (c.prop_use_dt) {
                    auto now = std::chrono::high_resolution_clock::now();
                    double d = std::chrono::duration<double>(now - last_dt_ts).count();
                    last_dt_ts = now;
                    return d;
                } else {
                    return std::max(0.001, 16.0 / 1000.0);
                }
            };
            double dt_pre = 0.0; // 按需赋值

            // sync control+mouse on main thread
            auto tctl0 = (c.timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{});
            double ctl_cost_ms = 0.0;
            double mou_cost_ms = 0.0;
            do {
                auto ds_copy = ds; // for pick helpers
                Detection* target = pick_target(ds_copy, c, ctr);

                if (!target) {
                    algo.reset();
                    break;
                }

                if (!smooth_pressed) { algo.reset(); }
                if (smooth_pressed && target && !single) {
                    cv::Point tp(target->box.x + target->box.width/2,
                                 target->box.y + (int)(target->box.height * c.y_anchor_ratio));
                    double err_x = (double)(tp.x - ctr.x);
                    double err_y = (double)(tp.y - ctr.y);
                    dt_pre = calc_dt();
                    auto pr = algo.step(err_x, err_y, dt_pre);
                    double ox = pr.first;
                    double oy = pr.second;
                    int mx = (int)(ox * mfac);
                    int my = (int)(oy * mfac);
                    auto tmov0 = (c.timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{});
                    if (mx || my) mouse.Move(mx, my);
                    auto tmov1 = (c.timing ? std::chrono::high_resolution_clock::now() : tmov0);
                    mou_cost_ms = c.timing ? std::chrono::duration<double, std::milli>(tmov1 - tmov0).count() : 0.0;
                }
                if (single && target) {
                    cv::Point tp(target->box.x + target->box.width/2,
                                 target->box.y + (int)(target->box.height * c.y_anchor_ratio));
                    double dx = (double)(tp.x - ctr.x);
                    double dy = (double)(tp.y - ctr.y);
                    int mx = (int)(dx * mfac);
                    int my = (int)(dy * mfac);
                    auto tmov0 = (c.timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{});
                    if (mx || my) mouse.Move(mx, my);
                    auto tmov1 = (c.timing ? std::chrono::high_resolution_clock::now() : tmov0);
                    mou_cost_ms = c.timing ? std::chrono::duration<double, std::milli>(tmov1 - tmov0).count() : 0.0;
                }
            } while(0);
            auto tctl1 = (c.timing ? std::chrono::high_resolution_clock::now() : tctl0);
            ctl_cost_ms = c.timing ? std::chrono::duration<double, std::milli>(tctl1 - tctl0).count() : 0.0;

            fl.mou_ms = mou_cost_ms;
            fl.ctl_ms = std::max(0.0, ctl_cost_ms - mou_cost_ms); // 去重：控制不含鼠标
            if (c.timing && c.ovl) { fl.ovl_ms = ovl.take_ms(); }

            auto loop_end_time = std::chrono::high_resolution_clock::now();
            double proc_ms = std::chrono::duration<double, std::milli>(loop_end_time - loop_start_time).count();
            if (c.fps_lock) {
                // FPS 锁基于主循环真实耗时，避免被其他线程阶段统计值放大而误限速
                double frame_ms = fc.end(proc_ms);
                tm.loop_ms = frame_ms;
            } else {
                tm.loop_ms = proc_ms;
            }
            // E2E≈max(loop,cap,det,io)。注意：ctl_ms 不含 mou_ms，避免重复计时
            if (c.timing) { double loop_e2e = std::max({tm.loop_ms, fl.cap_ms, fl.det_ms, fl.mou_ms, fl.ctl_ms, fl.ovl_ms});
            fl.loop_ms = loop_e2e;
            e2e_ms_last = loop_e2e; }

           // CSV async
           if (c.timing && c.lat_csv) { latcsv.put(fl); }

            
            // dt 按需计算（在执行算法前获取），此处仅保留统计输出
            if (c.timing) stats();

            if (c.timing && c.log_latency) {
                lat.push(fl);
                // drop/ovr 节流
                ++fcnt;
                int n = std::max(1, c.log_latency_every_n);
                if (fcnt % (size_t)n == 0) {
                    size_t drop_now = fc.drop - drop_prev;
                    lg::o() << lg::fx2
                            << P_L "[fps] "
                            << "drop=" << drop_now
                            << " | ovr=" << fc.ovr << "ms"
                            << lg::endl << lg::df;
                    // vis cache
                    dropS = drop_now;
                    ovrS = fc.ovr;
                    drop_prev = fc.drop;
                }
            }
            // 主线程输出采集统计
            if (c.timing) cap.reporter.drain();

            
            if (c.timing && !tm.trt_throughput.empty()) {
                lg::ml(P_T, tm.trt_throughput, tm.trt_cpu_lat, tm.trt_gpu_lat);
                tm.trt_throughput.clear();
                tm.trt_cpu_lat.clear();
                tm.trt_gpu_lat.clear();
            }

            if (vis) {
                draw();
            }
            if (g_exit_req.load(std::memory_order_relaxed)) { lg::o() << lg::nl << P_I "Signal exit" << lg::endl; break; }
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

    void draw() {
        for (const auto& det : ds) {
            // 显示全部；跟踪"*"；当前目标红框
            cv::Scalar color = c.is_target(det.class_id) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            cv::rectangle(img, det.box, color, 2);
            std::string label = c.name_or_id(det.class_id) + (c.is_target(det.class_id) ? "*" : "") +
                                 " " + cv::format("%.2f", det.confidence);
            cv::putText(img, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
        if (tTot > 0) {
            // FPS≈E2E
            double e2e = (e2e_ms_last > 0.0 ? e2e_ms_last : tTot);
            double smoothed_fps = 1000.0 / e2e;
            std::ostringstream stats_stream;
            stats_stream << lg::fx1 << "FPS: " << smoothed_fps << lg::df;
            cv::putText(img, stats_stream.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
            // DET/DROP/OVR
            std::ostringstream line2;
            line2 << lg::fx1;
            if (det_ms_last > 0.0) {
                line2 << "DET: " << (1000.0 / det_ms_last) << " | ";
            }
            line2 << "DROP: " << dropS << " | OVR: " << ovrS << "ms" << lg::df;
            cv::putText(img, line2.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 220, 255), 2);
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
    // sync mouse+algo
    Mouse mouse;
    AimAlgo algo;
    HotkeyManager hk;
    const cv::Rect crop;
    const cv::Point ctr;
    double mfac = 1.0;
    cv::Mat img;
    std::vector<Detection> ds;
    Timing tm;
    bool vis;
    double det_ms_last = 0.0;
    double e2e_ms_last = 0.0;
    double tTot = 0.0;
    bool first = true;
    // latency agg
    LatAgg lat;

    // FPS
    FpsCtl fc;
    size_t fcnt = 0;      // throttle cnt
    size_t drop_prev = 0; // last drop sum
    size_t dropS = 0;     // vis cache
    double ovrS = 0.0;

    std::chrono::high_resolution_clock::time_point last_dt_ts = std::chrono::high_resolution_clock::now();

    OvlAsync ovl;

    // CSV
    LatCsvAsync latcsv;
};


std::atomic<bool> g_exit_req{false};
std::atomic<bool> g_exit_done{false};
static BOOL WINAPI CtrlHandler(DWORD type) {
    switch (type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
        g_exit_req.store(true, std::memory_order_relaxed);
        return TRUE;
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        g_exit_req.store(true, std::memory_order_relaxed);
        // wait up to ~2s for graceful exit (CSV flush)
        for (int i = 0; i < 200; ++i) { // ~2s
            if (g_exit_done.load(std::memory_order_relaxed)) return TRUE;
            Sleep(10);
        }
        return FALSE; // not finished in time -> let system terminate
    default:
        return FALSE;
    }
}

int main() {
    SetConsoleCtrlHandler(CtrlHandler, TRUE);

    Config boot_cfg;
    boot_cfg.validate();
    // prio/MMCSS/overlay setup
    disablePowerThrottle();
    
    SetProcPrioHigh();
    SetThreadPrioHighest();

    
   
    HANDLE hMmcss = nullptr; // MMCSS disabled

   // thread mem prio + disable boost
   if (boot_cfg.enable_thread_mem_prio) {
       SetThreadMemPrioNormal();
       SetProcPrioBoostDisabled();
   }

    // overlay (Game overlay) banner based on ovl flag
    metrics::SystemBootReporting::PrintOverlayBanner(boot_cfg, boot_cfg.ovl);

    metrics::SystemBootReporting::PrintSchedulingBanner(boot_cfg, hMmcss);

    try {
        Aim assistant;
        assistant.loop();
    }
    catch (const std::exception& e) {
        // MMCSS disabled; no revert needed
        lg::ml(P_F, std::string("Fatal: ") + e.what(), "Check issues.", "Press Enter to exit.");
        std::cin.get();
        g_exit_done.store(true, std::memory_order_relaxed);
        return -1;
    }
    lg::o() << lg::nl << P_I "Done" << lg::endl;
    // MMCSS disabled; no revert needed
    g_exit_done.store(true, std::memory_order_relaxed);
    return 0;
}
