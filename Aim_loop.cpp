#include "Aim.hpp"
#include "pch.hpp"
#include "sh.hpp"

#include "CtrlHandler.hpp"
#include "target_pick.hpp"

void Aim::loop() {
  while (true) {
    if (g_exit_req.load(std::memory_order_relaxed)) {
      break;
    }
    auto loop_start_time = tn();

    FrameLat fl;

    ScreenCapturer::CapturedView view;
    if (!capture(fl, view)) {
      continue;
    }

    if (!infer_view(view)) {
      continue;
    }
    double ovl_ms = 0.0;
    fl.det_ms = c.timing ? (tm.infer_ms + tm.post_ms) : 0.0;

    bool smooth_pressed = hk.SmoothPressed(), single = hk.SingleShotJustPressed();

    auto tctl0 = (c.timing ? tn() : tp{});
    double ctl_cost_ms = 0.0;
    double mou_cost_ms = 0.0;
    {
      const Detection *target = pick_target(ds, c, ctr);
      if (!target) {
        algo.reset();
        last_tp_valid_ = false;
        lock_ = false;
        aim_ok_ = false;
      } else {
        if (!smooth_pressed) {
          algo.reset();
          last_tp_valid_ = false;
        }

        { auto ap = aim_no_pred(*target); aim_pt_.x = ap.x; aim_pt_.y = ap.y; aim_ok_ = true; lock_rect_ = {(float)(crop.x + target->box.x), (float)(crop.y + target->box.y), (float)target->box.width, (float)target->box.height}; lock_ = smooth_pressed || single; }

        double dt = get_dt();

        if (single) {
          bool use_pred = c.aim_single_use_pred;
          auto [dx, dy] = pred_err(*target, dt, use_pred);
          int mx = (int)(dx * mfac), my = (int)(dy * mfac); mv(mx, my, mou_cost_ms);
        } else if (smooth_pressed) {
          double h = (double)target->box.height;
          set_seg(h);
          auto [ex, ey] = pred_err(*target, dt, true);
          auto pr = algo.step(ex, ey, dt);
          int mx = (int)(pr.first * mfac), my = (int)(pr.second * mfac);
          if (!c.wind_enable) mv(mx, my, mou_cost_ms); else mv_wind(mx, my, mou_cost_ms);
        }
      }
    }
    auto tctl1 = (c.timing ? tn() : tctl0);
    ctl_cost_ms = c.timing ? msd(tctl1 - tctl0).count() : 0.0;

    fl.mou_ms = mou_cost_ms;
    fl.ctl_ms = MX(0.0, ctl_cost_ms - mou_cost_ms);

    draw_ovl(ovl_ms);
    fl.ovl_ms = ovl_ms;

    auto loop_end_time = tn();
    double proc_ms = msd(loop_end_time - loop_start_time).count();
    if (c.fps_lock) {
      tm.loop_ms = fc.end();
    } else {
      tm.loop_ms = proc_ms;
    }
    if (c.timing) {
      double loop_e2e = std::max({tm.loop_ms, fl.cap_ms, fl.det_ms, fl.mou_ms, fl.ctl_ms, fl.ovl_ms});
      fl.loop_ms = loop_e2e;
    }

    log_lat(fl);
    if (c.timing)
      cap.reporter.drain();
    trt_log();

    if (g_exit_req.load(std::memory_order_relaxed)) {
      break;
    }
  }
}
