#include "Aim.hpp"
#include "ovl_util.hpp"
#include "aim_ovl.hpp"
#include "sh.hpp"
#include <cmath>
#include <algorithm>

void aim_draw_fov(Aim* self){
  const auto& c = self->c;
  if (!c.ovl || !c.fov_show) return;
  int vx,vy,vw,vh; GetVirtualScreen(vx,vy,vw,vh);
  float cx = vx + vw * 0.5f;
  float cy = vy + vh * 0.5f;
  float r  = (float)c.fov_px;
  if (r <= 0.0f) return;
  OverlayColor col = ARGB((uint8_t)c.fov_a, (uint8_t)c.fov_r, (uint8_t)c.fov_g, (uint8_t)c.fov_b);
  int seg = std::max(3, c.fov_segments);
  const float PI = 3.14159265358979323846f;
  float th = (float)c.fov_th;
  uint16_t ly = (uint16_t)c.fov_layer;
  float prevx = cx + r, prevy = cy;
  for (int i=1;i<=seg;++i){
    float ang = (2.0f*PI*i)/seg;
    float x = cx + r*std::cos(ang);
    float y = cy + r*std::sin(ang);
    self->gover.AddLineL(prevx, prevy, x, y, th, col, ly);
    prevx = x; prevy = y;
  }
}
