#include "Aim.hpp"
#include "Log.hpp"
#include <chrono>

using clk = std::chrono::high_resolution_clock;
using tp = clk::time_point;
using msd = std::chrono::duration<double, std::milli>;
inline tp tn() { return clk::now(); }

void Aim::draw_ovl(double &ms) {
  if (!c.ovl) {
    ms = 0.0;
    return;
  }
  auto t0 = (c.timing ? tn() : tp{});
  gover.BeginFrame();
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
      if (cid < (int)c.ovl_cr.size())
        rr = c.ovl_cr[cid];
      if (cid < (int)c.ovl_cg.size())
        gg = c.ovl_cg[cid];
      if (cid < (int)c.ovl_cb.size())
        bb = c.ovl_cb[cid];
    }
    auto col = ARGB((uint8_t)c.ovl_a, (uint8_t)rr, (uint8_t)gg, (uint8_t)bb);
    gover.AddRect({x, y, w, h}, col, (float)c.ovl_thick);
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
  auto t1 = (c.timing ? tn() : tp{});
  ms = c.timing ? msd(t1 - t0).count() : 0.0;
}
