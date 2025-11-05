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






#include "OverlayAnimator.hpp"
#include "Mouse.hpp"
#include "PidController.hpp"
#include "HotkeyManager.hpp"
#include "ScreenCapturer.hpp"
#include "CapAsync.hpp"
#include "DetAsync.hpp"
#include "TrkAsync.hpp"
#include "OvlAsync.hpp"
#include "LatCsvAsync.hpp"



// helper pickers cloned from CtlAsync
static Detection* pick_by_tid(std::vector<Detection>& ds, int tid) {
    if (tid < 0) return nullptr;
    for (auto& d : ds) if (d.track_id == tid) return &d;
    return nullptr;
}
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
        cap(c),
        det(c),
        trk(c),
        hk(c),
        crop((cap.w() - c.crop_size_px) / 2, (cap.h() - c.crop_size_px) / 2, c.crop_size_px, c.crop_size_px),
        ctr(c.crop_size_px / 2, c.crop_size_px / 2),
        vis(c.enable_visual),
        lat(static_cast<size_t>(std::max(1, c.log_latency_every_n))),
        fc(c),
        ovl(c),
        latcsv(c)
    {
        if (c.lat_csv) latcsv.start(c.lat_csv_path);

        if (cap.w() < c.crop_size_px || cap.h() < c.crop_size_px) {
            throw std::runtime_error("Screen resolution is smaller than configured crop_size_px.");
        }
        mfac = (c.hfov_deg / static_cast<double>(cap.w()))
            * (c.px_per_360 / 360.0);
        if (vis) {
            cv::namedWindow(c.window_title, cv::WINDOW_AUTOSIZE);
        }
        lg::o() << "Detect center " << c.crop_size_px << "x" << c.crop_size_px << lg::endl;
        cap.set_roi(crop);
        cap.start();
        det.start();
        if (c.trk) trk.start();
        // init PID
        px.setK(c.pid_kp_x, c.pid_ki_x, c.pid_kd_x);
        px.setLim(c.pid_output_limit, c.pid_integral_limit);
        px.setDtClamp(c.pid_dt_clamp, c.pid_dt_min_ms, c.pid_dt_max_ms);
        px.setAntiWindupFreeze(c.pid_aw_freeze);
        py.setK(c.pid_kp_y, c.pid_ki_y, c.pid_kd_y);
        py.setLim(c.pid_output_limit, c.pid_integral_limit);
        py.setDtClamp(c.pid_dt_clamp, c.pid_dt_min_ms, c.pid_dt_max_ms);
        py.setAntiWindupFreeze(c.pid_aw_freeze);
        if (c.ovl) { lg::o() << P_B "Overlay=ON" << lg::endl; ovl.start(); }
        else { lg::o() << P_B "Overlay=OFF" << lg::endl; }
    }

    ~Aim() {
        if (c.lat_csv) latcsv.stop();
        cap.stop();
        det.stop();
        if (c.trk) trk.stop();
        if (c.ovl) ovl.stop();
        if (vis) cv::destroyAllWindows();
    }

    void loop() {
        while (true) {
            if (g_exit_req.load(std::memory_order_relaxed)) { lg::o() << lg::nl << P_I "Signal exit" << lg::endl; break; }
            // event-driven pacing: wait up to target frame time for new frame (not counted in proc_ms)
            int to = c.fps_lock ? (int)std::ceil(fc.tgt_ms) : 0;
            if (!cap.wait(to)) {
                if (g_exit_req.load(std::memory_order_relaxed)) { lg::o() << lg::nl << P_I "Signal exit" << lg::endl; break; }
                continue;
            }
            auto loop_start_time = std::chrono::high_resolution_clock::now();

            FrameLat fl;

            
            bool got = cap.get(img);
            if (!got) {
                // DXGI/WGC: 主线程不等待，避免双重计时
                fl.cap_ms = cap.last_ms();
                if (c.log_latency && !c.latency_on_capture_only) {
                    lat.push(fl);
                }
                if (g_exit_req.load(std::memory_order_relaxed)) { lg::o() << lg::nl << P_I "Signal exit" << lg::endl; break; }
                continue;
            }
            
            c.frame_idx++; fl.idx = c.frame_idx;
            fl.cap_ms = cap.last_ms();

            
            det.put(img);
            std::vector<Detection> ds_new;
            Timing tm_new;
            bool has_new = det.get(ds_new, tm_new);
            if (has_new) {
                ds = std::move(ds_new);
                tm = tm_new; // 覆盖检测阶段计时
                det_ms_last = tm.infer_ms + tm.post_ms;
                if (c.trk) {
                    trk.put(ds);
                    std::vector<Detection> ds_trk;
                    if (trk.get(ds_trk)) {
                        if (!ds_trk.empty()) ds = std::move(ds_trk); // only replace when tracker has confirmed outputs
                    }
                }
                if (c.ovl) ovl.put(ds, crop);
            }
            fl.det_ms = has_new ? det_ms_last : 0.0;

            
            
            bool smooth_pressed = hk.SmoothPressed();
            bool single = hk.SingleShotJustPressed();

            // 计算本帧 PID dt（控制前），避免上一帧dt滞后
            auto pid_now = std::chrono::high_resolution_clock::now();
            double dt_pre = c.pid_use_measured_dt
                ? std::chrono::duration<double>(pid_now - last_pid_ts).count()
                : std::max(0.001, c.pid_fixed_dt_ms / 1000.0);
            // 可选夹紧（主循环侧）
            if (c.pid_dt_clamp && c.pid_use_measured_dt) {
                double lo = std::max(0.000001, c.pid_dt_min_ms / 1000.0);
                double hi = std::max(lo, c.pid_dt_max_ms / 1000.0);
                if (dt_pre < lo) dt_pre = lo;
                if (dt_pre > hi) dt_pre = hi;
            }
            last_pid_ts = pid_now;

            // sync control+mouse on main thread
            auto tctl0 = std::chrono::high_resolution_clock::now();
            double ctl_cost_ms = 0.0;
            double mou_cost_ms = 0.0;
            do {
                auto ds_copy = ds; // for pick helpers
                Detection* target = pick_by_tid(ds_copy, cur_tid_);
                if (!target) target = pick_target(ds_copy, c, ctr);
                if (target && target->track_id >= 0) cur_tid_ = target->track_id;

                if (!target) {
                    px.reset(); py.reset();
                    if (!smooth_pressed || single) { break; }
                    if (!c.trk_pred_move) break;
                    Detection* pred_tgt = nullptr; double min_d = std::numeric_limits<double>::infinity();
                    for (auto& d : ds_copy) {
                        if (!d.is_pred) continue; if (!c.is_target(d.class_id)) continue;
                        cv::Point bc(d.box.x + d.box.width/2, d.box.y + (int)(d.box.height * c.y_anchor_ratio));
                        double dist = std::hypot(bc.x - ctr.x, bc.y - ctr.y);
                        if (dist < min_d) { min_d = dist; pred_tgt = &d; }
                    }
                    if (!pred_tgt) break;
                    cv::Point tp(pred_tgt->box.x + pred_tgt->box.width/2,
                                 pred_tgt->box.y + (int)(pred_tgt->box.height * c.y_anchor_ratio));
                    double err_x = (double)(tp.x - ctr.x);
                    double err_y = (double)(tp.y - ctr.y);
                    double ox = px.step(err_x, dt_pre);
                    double oy = py.step(err_y, dt_pre);
                    int mx = (int)(ox * mfac);
                    int my = (int)(oy * mfac);
                    auto tmov0 = std::chrono::high_resolution_clock::now();
                    if (mx || my) mouse.Move(mx, my);
                    auto tmov1 = std::chrono::high_resolution_clock::now();
                    mou_cost_ms = std::chrono::duration<double, std::milli>(tmov1 - tmov0).count();
                    break;
                }

                if (!smooth_pressed) { px.reset(); py.reset(); }
                if (smooth_pressed && target && !single) {
                    cv::Point tp(target->box.x + target->box.width/2,
                                 target->box.y + (int)(target->box.height * c.y_anchor_ratio));
                    double err_x = (double)(tp.x - ctr.x);
                    double err_y = (double)(tp.y - ctr.y);
                    double ox = px.step(err_x, dt_pre);
                    double oy = py.step(err_y, dt_pre);
                    int mx = (int)(ox * mfac);
                    int my = (int)(oy * mfac);
                    auto tmov0 = std::chrono::high_resolution_clock::now();
                    if (mx || my) mouse.Move(mx, my);
                    auto tmov1 = std::chrono::high_resolution_clock::now();
                    mou_cost_ms = std::chrono::duration<double, std::milli>(tmov1 - tmov0).count();
                }
                if (single && target) {
                    cv::Point tp(target->box.x + target->box.width/2,
                                 target->box.y + (int)(target->box.height * c.y_anchor_ratio));
                    double dx = (double)(tp.x - ctr.x);
                    double dy = (double)(tp.y - ctr.y);
                    int mx = (int)(dx * mfac);
                    int my = (int)(dy * mfac);
                    auto tmov0 = std::chrono::high_resolution_clock::now();
                    if (mx || my) mouse.Move(mx, my);
                    auto tmov1 = std::chrono::high_resolution_clock::now();
                    mou_cost_ms = std::chrono::duration<double, std::milli>(tmov1 - tmov0).count();
                }
            } while(0);
            auto tctl1 = std::chrono::high_resolution_clock::now();
            ctl_cost_ms = std::chrono::duration<double, std::milli>(tctl1 - tctl0).count();

            fl.mou_ms = mou_cost_ms;
            fl.ctl_ms = std::max(0.0, ctl_cost_ms - mou_cost_ms); // 去重：控制不含鼠标
            if (c.trk) { fl.trk_ms = trk.take_ms(); }
            if (c.ovl) { fl.ovl_ms = ovl.take_ms(); }

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
            double loop_e2e = std::max({tm.loop_ms, fl.cap_ms, fl.det_ms, fl.mou_ms, fl.ctl_ms, fl.ovl_ms, fl.trk_ms});
            fl.loop_ms = loop_e2e;
            e2e_ms_last = loop_e2e;

           // CSV async
           if (c.lat_csv) { latcsv.put(fl); }

            
            // dt 已在控制前计算为 dt_pre，这里仅保留统计
            stats();

            if (c.log_latency) {
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
            cap.drain();

            
            if (!tm.trt_throughput.empty()) {
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
    CapAsync cap;
    DetAsync det;
    TrkAsync trk;
    // sync mouse+pid
    Mouse mouse;
    PidController px, py;
    int cur_tid_ = -1;
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

    std::chrono::high_resolution_clock::time_point last_pid_ts = std::chrono::high_resolution_clock::now();

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
    OverlayAnim overlay;
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

    // overlay anim if enabled
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
        // MMCSS disabled; no revert needed
        lg::ml(P_F, std::string("Fatal: ") + e.what(), "Check issues.", "Press Enter to exit.");
        std::cin.get();
        g_exit_done.store(true, std::memory_order_relaxed);
        return -1;
    }
    lg::o() << lg::nl << P_I "Done" << lg::endl;
    overlay.stop();
    // MMCSS disabled; no revert needed
    g_exit_done.store(true, std::memory_order_relaxed);
    return 0;
}
