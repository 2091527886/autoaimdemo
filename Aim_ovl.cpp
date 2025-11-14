#include "Aim.hpp"
#include "Log.hpp"
#include "sh.hpp"
#include "ovl_util.hpp"

void Aim::draw_ovl(double &ms) {
  if (!c.ovl) { ms = 0.0; return; }
  auto t0 = (c.timing ? tn() : tp{});
  gover.BeginFrame();
  // draw detections as before
  for (const auto &d : ds) {
    if (c.ovl_only >= 0 && d.class_id != c.ovl_only)
      continue;
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
    auto col = ARGB((uint8_t)c.ovl_a, (uint8_t)rr, (uint8_t)gg, (uint8_t)bb);
    gover.AddRect({x, y, w, h}, col, (float)c.ovl_thick);
    float fsz = (float)std::max(1, c.ovl_font_px);
    float ty = std::max(0.0f, y - fsz - 2.0f);
    std::string lbl = c.name_or_id(cid);
    std::wstring wlbl(lbl.begin(), lbl.end());
    if (!wlbl.empty()) {
      OverlayColor tcol = ARGB(255, (uint8_t)rr, (uint8_t)gg, (uint8_t)bb);
      if (!c.ovl_font_name.empty()) gover.AddText(x, ty, wlbl, fsz, tcol, c.ovl_font_name);
      else gover.AddText(x, ty, wlbl, fsz, tcol);
    }
  }

  // draw crosshair (configurable)
  if (c.cross_show) {
    // choose aim point: predicted aim if valid; else screen center
    float px = (float)ctr.x;
    float py = (float)ctr.y;
    if (aim_ok_) { px = aim_pt_.x; py = aim_pt_.y; }
    // clamp to virtual screen
    int vx, vy, vw, vh;
    GetVirtualScreen(vx, vy, vw, vh);
    px = std::max((float)vx, std::min((float)(vx + vw - 1), px));
    py = std::max((float)vy, std::min((float)(vy + vh - 1), py));
    // style
    const float s = (float)c.cross_size;
    const float th = (float)c.cross_th;
    OverlayColor col = ARGB((uint8_t)c.cross_a, (uint8_t)c.cross_r, (uint8_t)c.cross_g, (uint8_t)c.cross_b);
    // draw
    gover.FillRect({px - s, py - th * 0.5f, 2*s, th}, col);
    gover.FillRect({px - th * 0.5f, py - s, th, 2*s}, col);
  }

  // draw lock highlight
  if (lock_) {
    OverlayColor ycol = ARGB((uint8_t)c.lock_a, (uint8_t)c.lock_r, (uint8_t)c.lock_g, (uint8_t)c.lock_b);
    gover.AddRect(lock_rect_, ycol, (float)c.lock_th);
    if (c.lock_show_text) {
      // text above rect with clamp to screen top
      float fsz = (float)std::max(1, c.ovl_font_px);
      float ty = std::max(0.0f, lock_rect_.y - fsz - 2.0f);
      const std::wstring& wtxt = c.lock_text;
      if (!c.ovl_font_name.empty())
        gover.AddText(lock_rect_.x, ty, wtxt, fsz, ycol, c.ovl_font_name);
      else
        gover.AddText(lock_rect_.x, ty, wtxt, fsz, ycol);
    }
  }
  gover.EndFrame();
  gover.Tick();
  auto t1 = (c.timing ? tn() : tp{});
  ms = c.timing ? msd(t1 - t0).count() : 0.0;
}
