#include "Aim.hpp"
#include "sh.hpp"
void aim_draw_lock(Aim* self) {
  const auto& c = self->c;
  if (!self->lock_) return;
  OverlayColor ycol = ARGB((uint8_t)c.lock_a, (uint8_t)c.lock_r, (uint8_t)c.lock_g, (uint8_t)c.lock_b);
  // rectangle drawn in draw_ovl when locked; avoid duplicate box here
  if (c.lock_show_text) {
    float fsz = (float)std::max(1, c.ovl_font_px);
    float ty = MX(0.0f, self->lock_rect_.y - fsz - 2.0f);
    const std::wstring& wtxt = c.lock_text;
    if (!c.ovl_font_name.empty())
      self->gover.AddText(self->lock_rect_.x, ty, wtxt, fsz, ycol, c.ovl_font_name);
    else
      self->gover.AddText(self->lock_rect_.x, ty, wtxt, fsz, ycol);
  }
}
