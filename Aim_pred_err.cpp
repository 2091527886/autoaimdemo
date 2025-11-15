#include "Aim.hpp"
#include "pch.hpp"
#include "sh.hpp"

using P2 = cv::Point2d;
using P2f = cv::Point2f;
std::pair<double, double> Aim::pred_err(const Detection &d, double dt, bool enable_pred) {
  double h = (double)d.box.height;
  double ya = anch(h);
  P2 tp(d.box.x + d.box.width / 2.0,
        d.box.y + d.box.height * ya);
  auto pred_step = [&](P2 &v_lp, const P2 &last_tp, P2 &tp_io) {
    double tc = MX(0.0, c.pred_v_tc_ms) / 1000.0;
    double a = (tc > 0.0) ? (dt / (tc + dt)) : 1.0;
    double inv_dt = 1.0 / MX(1e-6, dt);
    P2 v_inst{(tp_io.x - last_tp.x) * inv_dt, (tp_io.y - last_tp.y) * inv_dt};
    v_lp.x = v_lp.x + a * (v_inst.x - v_lp.x);
    v_lp.y = v_lp.y + a * (v_inst.y - v_lp.y);
    double lim = c.pred_v_lim_px;
    v_lp.x = MX(-lim, MN(lim, v_lp.x));
    v_lp.y = MX(-lim, MN(lim, v_lp.y));
    double ext = MN(dt, MX(0.0, c.pred_dt_max_ms) / 1000.0);
    tp_io.x += v_lp.x * ext;
    tp_io.y += v_lp.y * ext;
  };
  if (enable_pred && c.pred_enable && last_tp_valid_) {
    pred_step(v_lp_, last_tp_, tp);
  }
  last_tp_ = tp;
  last_tp_valid_ = true;
  return {tp.x - ctr.x, tp.y - ctr.y};
}

cv::Point2f Aim::aim_no_pred(const Detection &d) const {
  double ya = anch_read((double)d.box.height);
  double cx = crop.x + d.box.x + d.box.width / 2.0;
  double cy = crop.y + d.box.y + d.box.height * ya;
  return P2f((float)cx, (float)cy);
}
