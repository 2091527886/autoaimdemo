#include "Aim.hpp"
#include "pch.hpp"
#include "sh.hpp"
#include "ovl_util.hpp"
using P2 = cv::Point2d;
using P2f = cv::Point2f;

bool Aim::fov_ok(float x, float y) const {
  int vx,vy,vw,vh; GetVirtualScreen(vx,vy,vw,vh);
  float cx = vx + vw * 0.5f;
  float cy = vy + vh * 0.5f;
  float dx = x - cx;
  float dy = y - cy;
  float r = (float)c.fov_px;
  return (r <= 0.0f) ? true : (dx*dx + dy*dy <= r*r);
}

std::pair<double, double> Aim::pred_err(const Detection &d) {
  cv::Point2f ap = aim_no_pred(d);
  P2 tp(ap.x - crop.x, ap.y - crop.y);
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
