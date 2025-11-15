#include "Aim.hpp"
#include "pch.hpp"
#include "sh.hpp"

double Aim::anch(double h) {
  if (!c.aim_dyn_anchor)
    return c.y_anchor_ratio;
  double near_h = c.dist_near_h, far_h = c.dist_far_h, hys = c.aim_hys_h;
  last_seg_ = upd_seg(last_seg_, h, near_h, far_h, hys);
  return (last_seg_ == 2) ? c.y_anchor_head : c.y_anchor_body;
}

double Aim::anch_read(double h) const {
  if (!c.aim_dyn_anchor) return c.y_anchor_ratio;
  double near_h = c.dist_near_h, far_h = c.dist_far_h, hys = c.aim_hys_h;
  int seg = upd_seg(last_seg_, h, near_h, far_h, hys);
  return (seg == 2) ? c.y_anchor_head : c.y_anchor_body;
}

void Aim::set_seg(double h) {
  double near_h = c.dist_near_h, far_h = c.dist_far_h, hys = c.dist_hys_h;
  int seg = upd_seg(last_seg_, h, near_h, far_h, hys);
  last_seg_ = seg;
  auto setp = [&](double kx, double ky, double tc, double lim) { algo.setParams(kx, ky, tc, lim); };
  if (!c.curve_interp) {
    if (seg == 2)
      setp(c.seg_near_kx, c.seg_near_ky, c.seg_near_tc_ms, c.seg_near_lim);
    else if (seg == 1)
      setp(c.seg_mid_kx, c.seg_mid_ky, c.seg_mid_tc_ms, c.seg_mid_lim);
    else
      setp(c.seg_far_kx, c.seg_far_ky, c.seg_far_tc_ms, c.seg_far_lim);
  } else {
    double m = (far_h + near_h) * 0.5;
    if (h <= m) {
      double t = (h - far_h) / MX(1e-6, (m - far_h));
      t = CLP(t, 0.0, 1.0);
      setp(lerp(c.seg_far_kx, c.seg_mid_kx, t), lerp(c.seg_far_ky, c.seg_mid_ky, t),
           lerp(c.seg_far_tc_ms, c.seg_mid_tc_ms, t), lerp(c.seg_far_lim, c.seg_mid_lim, t));
    } else {
      double t = (h - m) / MX(1e-6, (near_h - m));
      t = CLP(t, 0.0, 1.0);
      setp(lerp(c.seg_mid_kx, c.seg_near_kx, t), lerp(c.seg_mid_ky, c.seg_near_ky, t),
           lerp(c.seg_mid_tc_ms, c.seg_near_tc_ms, t), lerp(c.seg_mid_lim, c.seg_near_lim, t));
    }
  }
}
