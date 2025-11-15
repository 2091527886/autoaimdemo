#include "Aim.hpp"
#include "pch.hpp"
#include "sh.hpp"

using P2 = cv::Point2d;
inline double msf(tp a, tp b) { return msd(a - b).count(); }

void Aim::mv(int mx, int my, double &ms) {
  if (mx || my) {
    auto t0 = (c.timing ? tn() : tp{});
    mouse.Move(mx, my);
    auto t1 = (c.timing ? tn() : t0);
    ms = c.timing ? msd(t1 - t0).count() : 0.0;
  }
}

void Aim::mv_wind(int mx, int my, double &ms) {
  auto t0 = (c.timing ? tn() : tp{});
  P2 want{(double)mx, (double)my};
  want.x += wind_.residual.x;
  want.y += wind_.residual.y;

  double seg_scale = 1.0;
  if (c.wind_use_seg_scale){ seg_scale = (last_seg_==2? c.wind_near_scale : (last_seg_==1? c.wind_mid_scale : c.wind_far_scale)); }
  double step_min = MX(0.0, c.wind_step_min_px * seg_scale);
  double step_max = MX(step_min, c.wind_step_max_px * seg_scale);
  double wind_k = c.wind_wind_k * seg_scale;
  double grav_k = c.wind_gravity_k * seg_scale;
  double jitter_s = c.wind_jitter_std_px * seg_scale;
  double end_slow = MX(0.0, c.wind_end_slow_px * seg_scale);
  double out_lim = MX(0.0, c.wind_out_limit_px);
  double dead_px = MX(0.0, c.wind_dead_px * seg_scale);

  auto budget_end = tn() + std::chrono::duration_cast<clk::duration>(msd(MX(0.0, c.wind_time_budget_ms)));
  int steps = 0;
  double sent = 0.0;
  const int steps_cap = std::max(1, c.wind_max_steps_per_frame);

  while (steps < steps_cap) {
    auto now = tn();
    if (now >= budget_end)
      break;
    double dist = hypot(want.x, want.y);
    if (dist <= dead_px)
      break;

    double ux = (dist > 1e-6) ? (want.x / dist) : 0.0;
    double uy = (dist > 1e-6) ? (want.y / dist) : 0.0;
    double tx = -uy, ty = ux;

    double t = CLP(1.0 - dist / MX(1.0, end_slow * 4.0), 0.0, 1.0);
    double curve = CLP(c.wind_speed_curve, 0.0, 1.0);
    double bell = (1.0 - curve) * t + curve * (1.0 - fabs(2.0 * t - 1.0));
    bell = CLP(bell, 0.1, 1.0);
    double step_len = CLP(dist * 0.25 * bell, step_min, step_max);

    double jx = wind_.n01(wind_.rng) * jitter_s;
    double jy = wind_.n01(wind_.rng) * jitter_s;

    double sx = grav_k * ux + wind_k * tx + jx;
    double sy = grav_k * uy + wind_k * ty + jy;
    double s_norm = HYP(sx, sy);
    if (s_norm < 1e-6) {
      sx = ux;
      sy = uy;
      s_norm = 1.0;
    }
    sx = sx / s_norm * step_len;
    sy = sy / s_norm * step_len;

    if (dist < end_slow) {
      sx -= c.wind_backtrack_k * (want.x - sx);
      sy -= c.wind_backtrack_k * (want.y - sy);
    }

    int dx = (int)LR(sx);
    int dy = (int)LR(sy);
    if (dx == 0 && dy == 0) {
      dx = (want.x > 0) ? 1 : ((want.x < 0) ? -1 : 0);
      dy = (want.y > 0) ? 1 : ((want.y < 0) ? -1 : 0);
      if (dx == 0 && dy == 0)
        break;
    }

    mouse.Move(dx, dy);
    want.x -= dx;
    want.y -= dy;
    sent += HYP((double)dx, (double)dy);
    ++steps;
    if (sent >= out_lim)
      break;
  }

  if (c.wind_target_noise_enable) {
    if (HYP(want.x, want.y) <= dead_px && c.wind_target_noise_px > 0.0) {
      int ndx = (int)LR(wind_.n01(wind_.rng) * c.wind_target_noise_px * 0.5);
      int ndy = (int)LR(wind_.n01(wind_.rng) * c.wind_target_noise_px * 0.5);
      if (ndx || ndy)
        mouse.Move(ndx, ndy);
    }
  }

  wind_.residual.x = want.x * c.wind_residual_carry;
  wind_.residual.y = want.y * c.wind_residual_carry;

  auto t1 = (c.timing ? tn() : t0);
  ms = c.timing ? msd(t1 - t0).count() : 0.0;
}
