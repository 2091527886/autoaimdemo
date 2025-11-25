#include "Aim.hpp"
#include "aim_ovl.hpp"
#include "Game_overlay.h"
#include "sh.hpp"

// 画调试中心：控制用中心(绿) vs 跟踪中心(红)
void aim_draw_dbg(Aim* self){
  const auto& c = self->c;
  if (!c.ovl || !c.dbg_centers) return;
  if (!self->dbg_has_pts_) return;
  auto dot = [&](float x,float y,uint32_t col){
    float s=2.0f, th=1.5f; uint16_t ly=(uint16_t)c.ovl_layer_dbg;
    self->gover.AddLineL(x-s,y, x+s,y, th, col, ly);
    self->gover.AddLineL(x, y-s, x, y+s, th, col, ly);
  };
  OverlayColor col_trk = ARGB(255,255,64,64);
  OverlayColor col_ctrl= ARGB(255,64,255,64);
  dot((float)self->dbg_trk_pt_.x,  (float)self->dbg_trk_pt_.y,  col_trk);
  dot((float)self->dbg_ctrl_pt_.x, (float)self->dbg_ctrl_pt_.y, col_ctrl);
}
