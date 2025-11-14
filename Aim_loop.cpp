#include "Aim.hpp"
#include "Log.hpp"
#include "pch.hpp"
#include "sh.hpp"

using std::clamp;
using std::max;
using std::min;

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

    bool smooth_pressed = hk.SmoothPressed();
    bool single = hk.SingleShotJustPressed();

    auto tctl0 = (c.timing ? tn() : tp{});
    double ctl_cost_ms = 0.0;
    double mou_cost_ms = 0.0;
    {
      const Detection *target = pick_target(ds, c, ctr);
      if (!target) {
        algo.reset();
        last_tp_valid_ = false;
      } else {
        if (!smooth_pressed) {
          algo.reset();
          last_tp_valid_ = false;
        }

        if (single) {
          double dt = get_dt();
          bool use_pred = c.aim_single_use_pred;
          auto [dx, dy] = pred_err(*target, dt, use_pred);
          int mx = (int)(dx * mfac);
          int my = (int)(dy * mfac);
          mv(mx, my, mou_cost_ms);
        } else if (smooth_pressed) {
          double dt = get_dt();
          double h = (double)target->box.height;
          set_seg(h);
          auto [ex, ey] = pred_err(*target, dt, true);
          auto pr = algo.step(ex, ey, dt);
          int mx = (int)(pr.first * mfac);
          int my = (int)(pr.second * mfac);
          if (!c.wind_enable)
            mv(mx, my, mou_cost_ms);
          else
            mv_wind(mx, my, mou_cost_ms);
        }
      }
    }
    auto tctl1 = (c.timing ? tn() : tctl0);
    ctl_cost_ms = c.timing ? msd(tctl1 - tctl0).count() : 0.0;

    fl.mou_ms = mou_cost_ms;
    fl.ctl_ms = std::max(0.0, ctl_cost_ms - mou_cost_ms);

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
