#include <iostream>
#include "Log.hpp"
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <sstream>
#include <thread>
#include <atomic>
#include <windows.h>

extern std::atomic<bool> g_exit_req;
extern std::atomic<bool> g_exit_done;

#include <opencv2/core.hpp>
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
#include "Game_overlay.h"


// helpers
static Detection* pick_target(const std::vector<Detection>& ds, const Config& c, const cv::Point& ctr) {
    Detection* target = nullptr;
    double min_dist_to_center = std::numeric_limits<double>::infinity();
    for (const auto& det : ds) {
        if (!c.is_target(det.class_id)) continue;
        cv::Point box_center(det.box.x + det.box.width / 2, det.box.y + det.box.height / 2);
        double dist = std::hypot(box_center.x - ctr.x, box_center.y - ctr.y);
        if (dist < min_dist_to_center) { min_dist_to_center = dist; target = const_cast<Detection*>(&det); }
    }
    return target;
}

// 计算目标相对中心的误差（像素）。y 使用 y_anchor_ratio。
static inline std::pair<double,double> calc_err(const Detection& d, const Config& c, const cv::Point& ctr) {
    cv::Point tp(d.box.x + d.box.width/2,
                 d.box.y + static_cast<int>(d.box.height * c.y_anchor_ratio));
    return { double(tp.x - ctr.x), double(tp.y - ctr.y) };
}

class Aim {
public:
    // FPS 锁定/补偿控制（短名）
    struct FpsCtl {
        double tgt_ms = 8.0;
        size_t drop = 0;
        double ovr = 0.0;
        std::chrono::steady_clock::time_point next_tp{};
        std::chrono::steady_clock::duration per{};
        explicit FpsCtl(const Config& cfg) {
            tgt_ms = 1000.0 / std::max(1, cfg.fps);
            per = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double, std::milli>(tgt_ms));
            next_tp = std::chrono::steady_clock::now() + per;
        }
        double end() {
            using clk = std::chrono::steady_clock;
            auto now = clk::now();
            if (now < next_tp) {
                std::this_thread::sleep_until(next_tp);
                ovr = 0.0;
                next_tp += per;
                return tgt_ms;
            }
            // lagging: compute overshoot and catch up phase without sleeping
            ovr = std::chrono::duration<double, std::milli>(now - next_tp).count();
            while (next_tp <= now) { next_tp += per; ++drop; }
            return tgt_ms;
        }
    };


    Aim()
        : c(),
        cap(c, c.crop_size_px, c.crop_size_px),
        od(c),
        hk(c),
        crop((cap.w() - c.crop_size_px) / 2, (cap.h() - c.crop_size_px) / 2, c.crop_size_px, c.crop_size_px),
        ctr(c.crop_size_px / 2, c.crop_size_px / 2),
        lat(static_cast<size_t>(std::max(1, c.log_latency_every_n))),
        fc(c)
    {

        if (cap.w() < c.crop_size_px || cap.h() < c.crop_size_px) {
            throw std::runtime_error("Screen resolution is smaller than configured crop_size_px.");
        }
        mfac = (c.hfov_deg / static_cast<double>(cap.w()))
            * (c.px_per_360 / 360.0);
        lg::o() << "Detect center " << c.crop_size_px << "x" << c.crop_size_px << lg::endl;
        // init AimAlgo (PropAlgo-based)
        algo.setConfig(c);
        // overlay moved to main thread
        if (c.ovl) {
            lg::o() << P_B "Overlay=ON (a=" << c.ovl_a << ", rgb=" << c.ovl_r << "," << c.ovl_g << "," << c.ovl_b
                    << ", th=" << c.ovl_thick << ")" << lg::endl;
            gover.UseVirtualScreen();
            gover.SetConfig(c);
            bool ok = gover.Start();
            lg::ok_or_werr(P_B, "Overlay.Start", ok);
        } else {
            lg::o() << P_B "Overlay=OFF" << lg::endl;
        }
    }

    ~Aim() {
        /* inference on main thread: no det.stop() */
        
        if (c.ovl) gover.Stop();
    }

    void loop() {
        while (true) {
            if (g_exit_req.load(std::memory_order_relaxed)) { break; }
            // no event pacing; capture on main thread
            auto loop_start_time = std::chrono::high_resolution_clock::now();

            FrameLat fl;

            // capture on main thread
            double cap_ms = 0.0;
            bool got = false;
            ScreenCapturer::CapturedView view;
            if (c.timing) {
                auto tcap0 = std::chrono::high_resolution_clock::now();
                got = cap.capture_view(view, crop);
                auto tcap1 = std::chrono::high_resolution_clock::now();
                cap_ms = std::chrono::duration<double, std::milli>(tcap1 - tcap0).count();
            } else {
                got = cap.capture_view(view, crop);
            }
            // no fallback clone path: strictly zero-copy view
            
            if (!got) {
                fl.cap_ms = cap_ms;
                if (c.timing && c.log_latency && !c.latency_on_capture_only) {
                    lat.push(fl);
                }
                continue;
            }
            c.frame_idx++; fl.idx = c.frame_idx;
            fl.cap_ms = cap_ms;

            // inference on main thread (strict zero-copy)
            if (view.mapped && view.data) {
                od.detect_view(view.data, view.width, view.height, view.pitch, ds, c, tm);
                cap.release_view(view);
            } else {
                // unexpected: no mapped view; skip this frame to keep freshness
                cap.release_view(view);
                continue;
            }
            // overlay draw on main thread (moved below)
            double ovl_ms = 0.0;
            // [overlay moved below]
            fl.det_ms = c.timing ? (tm.infer_ms + tm.post_ms) : 0.0;

            
            
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

            // sync control+mouse on main thread
            auto tctl0 = (c.timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{});
            double ctl_cost_ms = 0.0;
            double mou_cost_ms = 0.0;
            {
                Detection* target = pick_target(ds, c, ctr);
                if (!target) { algo.reset(); last_tp_valid_ = false; }
                else {
                    if (!smooth_pressed) { algo.reset(); last_tp_valid_ = false; }

                    int mx = 0, my = 0;
                    auto decide_anchor = [&](double h)->double{
                        if (!c.aim_dyn_anchor) return c.y_anchor_ratio;
                        // 分段与滞回
                        double near_h = c.dist_near_h, far_h = c.dist_far_h, hys = c.aim_hys_h;
                        int seg = last_seg_;
                        if (seg == 2) { // NEAR -> 仅当明显变小才降级
                            if (h < near_h - hys) seg = 1;
                        } else if (seg == 0) { // FAR -> 仅当明显变大才升级
                            if (h > far_h + hys) seg = 1;
                        } else { // MID
                            if (h >= near_h) seg = 2;
                            else if (h <= far_h) seg = 0;
                        }
                        last_seg_ = seg; last_h_ = h;
                        // 近头远身，中间取身体
                        return (seg==2) ? c.y_anchor_head : c.y_anchor_body;
                    };

                    auto compute_pred_err = [&](const Detection& d, double dt, bool enable_pred){
                        double h = (double)d.box.height;
                        double ya = decide_anchor(h);
                        cv::Point2d tp(d.box.x + d.box.width/2.0,
                                       d.box.y + d.box.height * ya);
                        // 预测
                        if (enable_pred && c.pred_enable && last_tp_valid_) {
                            double tc = std::max(0.0, c.pred_v_tc_ms) / 1000.0;
                            double a = (tc>0.0) ? (dt/(tc+dt)) : 1.0;
                            double inv_dt = 1.0 / std::max(1e-6, dt);
                            cv::Point2d v_inst{ (tp.x - last_tp_.x) * inv_dt, (tp.y - last_tp_.y) * inv_dt };
                            v_lp_.x = v_lp_.x + a * (v_inst.x - v_lp_.x);
                            v_lp_.y = v_lp_.y + a * (v_inst.y - v_lp_.y);
                            // 限幅
                            double lim = c.pred_v_lim_px;
                            v_lp_.x = std::max(-lim, std::min(lim, v_lp_.x));
                            v_lp_.y = std::max(-lim, std::min(lim, v_lp_.y));
                            double ext = std::min(dt, std::max(0.0, c.pred_dt_max_ms)/1000.0);
                            tp.x += v_lp_.x * ext;
                            tp.y += v_lp_.y * ext;
                        }
                        last_tp_ = tp; last_tp_valid_ = true;
                        return std::pair<double,double>(tp.x - ctr.x, tp.y - ctr.y);
                    };

                    auto apply_seg_params = [&](double h){
                        // 选择段 + 可选插值
                        double near_h = c.dist_near_h, far_h = c.dist_far_h, hys = c.dist_hys_h;
                        int seg = last_seg_;
                        if (seg == 2) { if (h < near_h - hys) seg = 1; }
                        else if (seg == 0) { if (h > far_h + hys) seg = 1; }
                        else { if (h >= near_h) seg = 2; else if (h <= far_h) seg = 0; }
                        last_seg_ = seg; last_h_ = h;
                        auto setp=[&](double kx,double ky,double tc,double lim){ algo.setParams(kx,ky,tc,lim); };
                        if (!c.curve_interp) {
                            if (seg==2) setp(c.seg_near_kx,c.seg_near_ky,c.seg_near_tc_ms,c.seg_near_lim);
                            else if (seg==1) setp(c.seg_mid_kx,c.seg_mid_ky,c.seg_mid_tc_ms,c.seg_mid_lim);
                            else setp(c.seg_far_kx,c.seg_far_ky,c.seg_far_tc_ms,c.seg_far_lim);
                        } else {
                            double m = (far_h + near_h) * 0.5;
                            auto lerp=[&](double a,double b,double t){ return a + (b-a)*t; };
                            if (h <= m) {
                                double t = (h - far_h)/std::max(1e-6, (m - far_h)); t = std::clamp(t, 0.0, 1.0);
                                setp(lerp(c.seg_far_kx,c.seg_mid_kx,t), lerp(c.seg_far_ky,c.seg_mid_ky,t),
                                     lerp(c.seg_far_tc_ms,c.seg_mid_tc_ms,t), lerp(c.seg_far_lim,c.seg_mid_lim,t));
                            } else {
                                double t = (h - m)/std::max(1e-6, (near_h - m)); t = std::clamp(t, 0.0, 1.0);
                                setp(lerp(c.seg_mid_kx,c.seg_near_kx,t), lerp(c.seg_mid_ky,c.seg_near_ky,t),
                                     lerp(c.seg_mid_tc_ms,c.seg_near_tc_ms,t), lerp(c.seg_mid_lim,c.seg_near_lim,t));
                            }
                        }
                    };

                    if (single) {
                        double dt = calc_dt();
                        bool use_pred = c.aim_single_use_pred;
                        double h = (double)target->box.height;
                        auto [dx, dy] = compute_pred_err(*target, dt, use_pred);
                        mx = (int)(dx * mfac);
                        my = (int)(dy * mfac);
                    } else if (smooth_pressed) {
                        double dt = calc_dt();
                        // 分段参数
                        double h = (double)target->box.height;
                        apply_seg_params(h);
                        auto [ex, ey] = compute_pred_err(*target, dt, true);
                        auto pr = algo.step(ex, ey, dt);
                        mx = (int)(pr.first  * mfac);
                        my = (int)(pr.second * mfac);
                    }

                    if (mx || my) {
                        auto tmov0 = (c.timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{});
                        mouse.Move(mx, my);
                        auto tmov1 = (c.timing ? std::chrono::high_resolution_clock::now() : tmov0);
                        mou_cost_ms = c.timing ? std::chrono::duration<double, std::milli>(tmov1 - tmov0).count() : 0.0;
                    }
                }
            }
            auto tctl1 = (c.timing ? std::chrono::high_resolution_clock::now() : tctl0);
            ctl_cost_ms = c.timing ? std::chrono::duration<double, std::milli>(tctl1 - tctl0).count() : 0.0;

            fl.mou_ms = mou_cost_ms;
            fl.ctl_ms = std::max(0.0, ctl_cost_ms - mou_cost_ms); // 去重：控制不含鼠标

            // overlay draw on main thread (now after mouse movement)
            if (c.ovl) {
                auto t0 = (c.timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{});
                gover.BeginFrame();
                for (const auto& d : ds) {
                    if (c.ovl_only >= 0 && d.class_id != c.ovl_only) continue;
                    float x = (float)(crop.x + d.box.x);
                    float y = (float)(crop.y + d.box.y);
                    float w = (float)d.box.width;
                    float h = (float)d.box.height;
                    int cid = d.class_id;
                    int rr = c.ovl_r, gg = c.ovl_g, bb = c.ovl_b;
                    if (c.ovl_cls && cid >= 0) {
                        if (cid < (int)c.ovl_cr.size()) rr = c.ovl_cr[cid];
                        if (cid < (int)c.ovl_cg.size()) gg = c.ovl_cg[cid];
                        if (cid < (int)c.ovl_cb.size()) bb = c.ovl_cb[cid];
                    }
                    auto col = ARGB((uint8_t)c.ovl_a,(uint8_t)rr,(uint8_t)gg,(uint8_t)bb);
                    gover.AddRect({x,y,w,h}, col, (float)c.ovl_thick);
                }
                gover.EndFrame();
                gover.Tick();
                auto t1 = (c.timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{});
                ovl_ms = c.timing ? std::chrono::duration<double, std::milli>(t1 - t0).count() : 0.0;
            }
            fl.ovl_ms = ovl_ms;

            auto loop_end_time = std::chrono::high_resolution_clock::now();
            double proc_ms = std::chrono::duration<double, std::milli>(loop_end_time - loop_start_time).count();
            if (c.fps_lock) {
                // FPS 锁基于主循环真实耗时，避免被其他线程阶段统计值放大而误限速
                tm.loop_ms = fc.end();
            } else {
                tm.loop_ms = proc_ms;
            }
            // E2E≈max(loop,cap,det,io)。注意：ctl_ms 不含 mou_ms，避免重复计时
            if (c.timing) {
                double loop_e2e = std::max({tm.loop_ms, fl.cap_ms, fl.det_ms, fl.mou_ms, fl.ctl_ms, fl.ovl_ms});
                fl.loop_ms = loop_e2e;
            }

           // CSV async
            
            // dt 按需计算（在执行算法前获取）

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

            if (g_exit_req.load(std::memory_order_relaxed)) { break; }
        }
    }

private:
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
    // removed legacy CPU clone buffer
    // cv::Mat img;
    std::vector<Detection> ds;
    Timing tm;
    // latency agg
    LatAgg lat;

    // FPS
    FpsCtl fc;
    size_t fcnt = 0;      // throttle cnt
    size_t drop_prev = 0; // last drop sum

    std::chrono::high_resolution_clock::time_point last_dt_ts = std::chrono::high_resolution_clock::now();

    Game_overlay gover;

    // 预测/分段状态
    cv::Point2d v_lp_ {0.0, 0.0};
    cv::Point2d last_tp_ {0.0, 0.0};
    bool last_tp_valid_ = false;
    int  last_seg_ = 1; // 0:far,1:mid,2:near
    double last_h_ = 0.0;
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
        // wait up to ~2s for graceful exit
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

   // thread mem prio + disable boost
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
