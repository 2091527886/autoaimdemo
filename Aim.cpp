#include <iostream>
#include "Log.hpp"
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <thread>
#include <atomic>
#include <random>
#include <windows.h>

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
#include "target_pick.hpp"
#include "CtrlHandler.hpp"
#include "Aim.hpp"

Aim::Aim()
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
    // seed wind rng if needed
    if (c.wind_rng_seed > 0) {
        wind_.rng.seed(static_cast<uint32_t>(c.wind_rng_seed));
    }
    if (c.ovl) {
        gover.UseVirtualScreen();
        gover.SetConfig(c);
        bool ok = gover.Start();
        lg::ok_or_werr(P_B, "Overlay.Start", ok);
    }
}

Aim::~Aim() {
    if (c.ovl) gover.Stop();
}

void Aim::loop() {
    while (true) {
        if (g_exit_req.load(std::memory_order_relaxed)) { break; }
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
            // unexpected: no mapped view; skip this frame
            cap.release_view(view);
            continue;
        }
        double ovl_ms = 0.0;
        fl.det_ms = c.timing ? (tm.infer_ms + tm.post_ms) : 0.0;

        bool smooth_pressed = hk.SmoothPressed();
        bool single = hk.SingleShotJustPressed();

        // lazy dt
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

        auto tctl0 = (c.timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{});
        double ctl_cost_ms = 0.0;
        double mou_cost_ms = 0.0;
        {
            const Detection* target = pick_target(ds, c, ctr);
            if (!target) { algo.reset(); last_tp_valid_ = false; }
            else {
                if (!smooth_pressed) { algo.reset(); last_tp_valid_ = false; }

                int mx = 0, my = 0;
                auto upd_seg = [&](int seg, double h, double near_h, double far_h, double hys){
                    if (seg == 2) { if (h < near_h - hys) seg = 1; }
                    else if (seg == 0) { if (h > far_h + hys) seg = 1; }
                    else { if (h >= near_h) seg = 2; else if (h <= far_h) seg = 0; }
                    return seg;
                };
                auto decide_anchor = [&](double h)->double{
                    if (!c.aim_dyn_anchor) return c.y_anchor_ratio;
                    double near_h = c.dist_near_h, far_h = c.dist_far_h, hys = c.aim_hys_h;
                    last_seg_ = upd_seg(last_seg_, h, near_h, far_h, hys);
                    return (last_seg_==2) ? c.y_anchor_head : c.y_anchor_body;
                };

                auto compute_pred_err = [&](const Detection& d, double dt, bool enable_pred){
                    double h = (double)d.box.height;
                    double ya = decide_anchor(h);
                    cv::Point2d tp(d.box.x + d.box.width/2.0,
                                   d.box.y + d.box.height * ya);
                    auto pred_step = [&](cv::Point2d& v_lp, const cv::Point2d& last_tp, cv::Point2d& tp_io){
                        double tc = std::max(0.0, c.pred_v_tc_ms) / 1000.0;
                        double a = (tc>0.0) ? (dt/(tc+dt)) : 1.0;
                        double inv_dt = 1.0 / std::max(1e-6, dt);
                        cv::Point2d v_inst{ (tp_io.x - last_tp.x) * inv_dt, (tp_io.y - last_tp.y) * inv_dt };
                        v_lp.x = v_lp.x + a * (v_inst.x - v_lp.x);
                        v_lp.y = v_lp.y + a * (v_inst.y - v_lp.y);
                        double lim = c.pred_v_lim_px;
                        v_lp.x = std::max(-lim, std::min(lim, v_lp.x));
                        v_lp.y = std::max(-lim, std::min(lim, v_lp.y));
                        double ext = std::min(dt, std::max(0.0, c.pred_dt_max_ms)/1000.0);
                        tp_io.x += v_lp.x * ext;
                        tp_io.y += v_lp.y * ext;
                    };
                    if (enable_pred && c.pred_enable && last_tp_valid_) {
                        pred_step(v_lp_, last_tp_, tp);
                    }
                    last_tp_ = tp; last_tp_valid_ = true;
                    return std::pair<double,double>(tp.x - ctr.x, tp.y - ctr.y);
                };

                auto apply_seg_params = [&](double h){
                    double near_h = c.dist_near_h, far_h = c.dist_far_h, hys = c.dist_hys_h;
                    int seg = last_seg_;
                    if (seg == 2) { if (h < near_h - hys) seg = 1; }
                    else if (seg == 0) { if (h > far_h + hys) seg = 1; }
                    else { if (h >= near_h) seg = 2; else if (h <= far_h) seg = 0; }
                    last_seg_ = seg;
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
                    auto [dx, dy] = compute_pred_err(*target, dt, use_pred);
                    int mx = (int)(dx * mfac);
                    int my = (int)(dy * mfac);
                    if (mx || my) {
                        auto tmov0 = (c.timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{});
                        mouse.Move(mx, my);
                        auto tmov1 = (c.timing ? std::chrono::high_resolution_clock::now() : tmov0);
                        mou_cost_ms = c.timing ? std::chrono::duration<double, std::milli>(tmov1 - tmov0).count() : 0.0;
                    }
                } else if (smooth_pressed) {
                    double dt = calc_dt();
                    double h = (double)target->box.height;
                    apply_seg_params(h);
                    auto [ex, ey] = compute_pred_err(*target, dt, true);
                    auto pr = algo.step(ex, ey, dt);
                    int mx = (int)(pr.first  * mfac);
                    int my = (int)(pr.second * mfac);
                    if (mx || my) {
                        auto tmov0 = (c.timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{});
                        if (!c.wind_enable) {
                            mouse.Move(mx, my);
                        } else {
                            cv::Point2d want{ (double)mx, (double)my };
                            want.x += wind_.residual.x; want.y += wind_.residual.y;

                            double seg_scale = 1.0;
                            if (c.wind_use_seg_scale) {
                                if (last_seg_ == 2) seg_scale = c.wind_near_scale;
                                else if (last_seg_ == 1) seg_scale = c.wind_mid_scale;
                                else seg_scale = c.wind_far_scale;
                            }
                            double step_min = std::max(0.0, c.wind_step_min_px * seg_scale);
                            double step_max = std::max(step_min, c.wind_step_max_px * seg_scale);
                            double wind_k   = c.wind_wind_k * seg_scale;
                            double grav_k   = c.wind_gravity_k * seg_scale;
                            double jitter_s = c.wind_jitter_std_px * seg_scale;
                            double end_slow = std::max(0.0, c.wind_end_slow_px * seg_scale);
                            double out_lim  = std::max(0.0, c.wind_out_limit_px);
                            double dead_px  = std::max(0.0, c.wind_dead_px * seg_scale);

                            auto budget_end = std::chrono::high_resolution_clock::now()
                                + std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(
                                      std::chrono::duration<double, std::milli>(std::max(0.0, c.wind_time_budget_ms)));
                            int steps = 0;
                            double sent = 0.0;
                            const int steps_cap = std::max(1, c.wind_max_steps_per_frame);

                            while (steps < steps_cap) {
                                auto now = std::chrono::high_resolution_clock::now();
                                if (now >= budget_end) break;
                                double dist = std::hypot(want.x, want.y);
                                if (dist <= dead_px) break;

                                double ux = (dist > 1e-6) ? (want.x / dist) : 0.0;
                                double uy = (dist > 1e-6) ? (want.y / dist) : 0.0;
                                double tx = -uy, ty = ux;

                                double t = std::clamp(1.0 - dist / std::max(1.0, end_slow * 4.0), 0.0, 1.0);
                                double curve = std::clamp(c.wind_speed_curve, 0.0, 1.0);
                                double bell = (1.0 - curve) * t + curve * (1.0 - std::abs(2.0 * t - 1.0));
                                bell = std::clamp(bell, 0.1, 1.0);
                                double step_len = std::clamp(dist * 0.25 * bell, step_min, step_max);

                                double jx = wind_.n01(wind_.rng) * jitter_s;
                                double jy = wind_.n01(wind_.rng) * jitter_s;

                                double sx = grav_k * ux + wind_k * tx + jx;
                                double sy = grav_k * uy + wind_k * ty + jy;
                                double s_norm = std::hypot(sx, sy);
                                if (s_norm < 1e-6) { sx = ux; sy = uy; s_norm = 1.0; }
                                sx = sx / s_norm * step_len;
                                sy = sy / s_norm * step_len;

                                if (dist < end_slow) {
                                    sx -= c.wind_backtrack_k * (want.x - sx);
                                    sy -= c.wind_backtrack_k * (want.y - sy);
                                }

                                int dx = (int)std::lround(sx);
                                int dy = (int)std::lround(sy);
                                if (dx == 0 && dy == 0) {
                                    dx = (want.x > 0) ? 1 : ((want.x < 0) ? -1 : 0);
                                    dy = (want.y > 0) ? 1 : ((want.y < 0) ? -1 : 0);
                                    if (dx == 0 && dy == 0) break;
                                }

                                mouse.Move(dx, dy);
                                want.x -= dx; want.y -= dy;
                                sent += std::hypot((double)dx, (double)dy);
                                ++steps;
                                if (sent >= out_lim) break;
                            }

                            if (c.wind_target_noise_enable) {
                                if (std::hypot(want.x, want.y) <= dead_px && c.wind_target_noise_px > 0.0) {
                                    int ndx = (int)std::lround(wind_.n01(wind_.rng) * c.wind_target_noise_px * 0.5);
                                    int ndy = (int)std::lround(wind_.n01(wind_.rng) * c.wind_target_noise_px * 0.5);
                                    if (ndx || ndy) mouse.Move(ndx, ndy);
                                }
                            }

                            wind_.residual.x = want.x * c.wind_residual_carry;
                            wind_.residual.y = want.y * c.wind_residual_carry;
                        }
                        auto tmov1 = (c.timing ? std::chrono::high_resolution_clock::now() : tmov0);
                        mou_cost_ms = c.timing ? std::chrono::duration<double, std::milli>(tmov1 - tmov0).count() : 0.0;
                    }
                }
            }
        }
        auto tctl1 = (c.timing ? std::chrono::high_resolution_clock::now() : tctl0);
        ctl_cost_ms = c.timing ? std::chrono::duration<double, std::milli>(tctl1 - tctl0).count() : 0.0;

        fl.mou_ms = mou_cost_ms;
        fl.ctl_ms = std::max(0.0, ctl_cost_ms - mou_cost_ms);

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
                float fsz = (float)std::max(1, c.ovl_font_px);
                float ty = std::max(0.0f, y - fsz - 2.0f);
                std::string lbl = c.name_or_id(cid);
                std::wstring wlbl(lbl.begin(), lbl.end());
                if (!wlbl.empty()) {
                    OverlayColor tcol = ARGB(255, (uint8_t)rr, (uint8_t)gg, (uint8_t)bb);
                    if (!c.ovl_font_name.empty())
                        gover.AddText(x, ty, wlbl, fsz, tcol, c.ovl_font_name);
                    else
                        gover.AddText(x, ty, wlbl, fsz, tcol);
                }
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
            tm.loop_ms = fc.end();
        } else {
            tm.loop_ms = proc_ms;
        }
        if (c.timing) {
            double loop_e2e = std::max({tm.loop_ms, fl.cap_ms, fl.det_ms, fl.mou_ms, fl.ctl_ms, fl.ovl_ms});
            fl.loop_ms = loop_e2e;
        }

        if (c.timing && c.log_latency) {
            lat.push(fl);
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
