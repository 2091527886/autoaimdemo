#include "Aim.hpp"
#include "pch.hpp"
#include "sh.hpp"

using P2 = cv::Point2d;
using P2f = cv::Point2f;
std::pair<double, double> Aim::pred_err(const Detection &d, double dt, bool enable_pred) {
  // double h = (double)d.box.height; // unused
  // prefer pose point (same as aim_no_pred) as tracking target, fallback to bbox anchor
  cv::Point2f ap = aim_no_pred(d);
  P2 tp(ap.x - crop.x, ap.y - crop.y);
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

int Aim::det_index(const Detection &d) const {
  for (size_t i=0;i<ds.size();++i){ if (&ds[i] == &d) return (int)i; }
  return -1;
}

cv::Point2f Aim::aim_no_pred(const Detection &d) const {
  // prefer pose points when available; fallback to bbox anchor
  int idx = det_index(d);
  if (idx>=0 && idx < (int)pose_ok.size() && pose_ok[idx] && idx < (int)pose_kpts.size()) {
    const auto& k = pose_kpts[idx];
    auto okp = [&](int i, cv::Point2f& p){ if (i<(int)k.size() && (!k[i].conf || *k[i].conf >= c.pose_kpt_conf_thresh)) { p = cv::Point2f((float)(crop.x + k[i].x), (float)(crop.y + k[i].y)); return true; } return false; };
    auto aim_nose = [&]() -> std::optional<P2f>{ cv::Point2f pn; if (okp(0,pn)) return pn; return std::optional<P2f>(); };
    auto aim_chest = [&]() -> std::optional<P2f>{ cv::Point2f p5,p6,p12,p11; bool a=okp(5,p5), b=okp(6,p6), c1=okp(12,p12), d1=okp(11,p11); if (a&&b&&c1&&d1){ float cx=(p5.x+p6.x+p12.x+p11.x)*0.25f; float cy=(p5.y+p6.y+p12.y+p11.y)*0.25f; return P2f(cx,cy); } return std::optional<P2f>(); };
    if (c.pose_aim_pref_nose){ if (auto r=aim_nose()) return *r; if (auto r=aim_chest()) return *r; }
    else { if (auto r=aim_chest()) return *r; if (auto r=aim_nose()) return *r; }
  }
  double ya = anch_read((double)d.box.height);
  double cx = crop.x + d.box.x + d.box.width / 2.0;
  double cy = crop.y + d.box.y + d.box.height * ya;
  return P2f((float)cx, (float)cy);
}
