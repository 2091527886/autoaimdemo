#include "Aim.hpp"
#include "ovl_util.hpp"
#include "sh.hpp"
void aim_draw_cross(Aim* self) {
  const auto& c = self->c;
  if (!c.cross_show) return;
  float px = (float)self->ctr.x;
  float py = (float)self->ctr.y;
  if (self->aim_ok_) { px = self->aim_pt_.x; py = self->aim_pt_.y; }
  int vx, vy, vw, vh; GetVirtualScreen(vx, vy, vw, vh);
  px = MX((float)vx, MN((float)(vx + vw - 1), px));
  py = MX((float)vy, MN((float)(vy + vh - 1), py));
  const float s = (float)c.cross_size;
  const float th = (float)c.cross_th;
  OverlayColor col = ARGB((uint8_t)c.cross_a, (uint8_t)c.cross_r, (uint8_t)c.cross_g, (uint8_t)c.cross_b);
  self->gover.FillRect({px - s, py - th * 0.5f, 2*s, th}, col);
  self->gover.FillRect({px - th * 0.5f, py - s, th, 2*s}, col);
}
