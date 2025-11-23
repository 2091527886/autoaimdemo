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

void Aim::mv_precise(double wx, double wy, double &ms) {
  // 非风路径的亚像素残差累积，避免小输出被截断为0
  auto t0 = (c.timing ? tn() : tp{});
  double want_x = wx + mv_residual_.x;
  double want_y = wy + mv_residual_.y;
  int dx = (int)LR(want_x);
  int dy = (int)LR(want_y);
  if (dx || dy) {
    mouse.Move(dx, dy);
  }
  // 调试：统计“被量化吞掉”的小输出（更精确：每帧统计残差长度）
  {
    double loss = std::hypot(want_x - dx, want_y - dy);
    if (loss > 0) { mvq_loss_sum_ += loss; ++mvq_events_; }
  }
  // 残差按系数回填（0~1）
  double carry = CLP(c.aim_residual_carry, 0.0, 1.0);
  mv_residual_.x = (want_x - dx) * carry;
  mv_residual_.y = (want_y - dy) * carry;
  auto t1 = (c.timing ? tn() : t0);
  ms = c.timing ? msd(t1 - t0).count() : 0.0;
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
    auto now = tn(); if (now >= budget_end) break;
    double dist = HYP(want.x, want.y); if (dist <= dead_px) break;

    double ux, uy; unit(want.x, want.y, ux, uy); double tx = -uy, ty = ux;

    double t = CLP(1.0 - dist / MX(1.0, end_slow * 4.0), 0.0, 1.0);
    double bell = bell_curve(t, c.wind_speed_curve);
    double step_len = step_len_px(dist, bell, step_min, step_max);

    double jx = wind_.n01(wind_.rng) * jitter_s;
    double jy = wind_.n01(wind_.rng) * jitter_s;

    double sx = grav_k * ux + wind_k * tx + jx;
    double sy = grav_k * uy + wind_k * ty + jy;
    double s_norm = HYP(sx, sy);
    if (s_norm < 1e-6) { sx = ux; sy = uy; s_norm = 1.0; }
    sx = sx / s_norm * step_len; sy = sy / s_norm * step_len;

    if (dist < end_slow) { backtrack(sx, sy, want.x, want.y, c.wind_backtrack_k); }

    int dx = (int)LR(sx), dy = (int)LR(sy);
    minstep(dx, dy, want.x, want.y); if (dx == 0 && dy == 0) break;

    mouse.Move(dx, dy);
    want.x -= dx; want.y -= dy;
    sent += HYP((double)dx, (double)dy);
    ++steps; if (sent >= out_lim) break;
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
