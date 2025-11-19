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
    // 推入统一文本管道，优先级使用 3（与 PRI_LOCK 对齐）
    Aim::PendingText pt{ (float)self->lock_rect_.x, ty, wtxt, ycol, (uint16_t)c.ovl_layer_lock, 3, 0 };
    self->pending_texts_.push_back(std::move(pt));
  }
}
