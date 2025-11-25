#include "Aim.hpp"
#include "ovl_util.hpp"
#include "aim_ovl.hpp"
#include "sh.hpp"
#include <cmath>
#include <algorithm>

void aim_draw_dz(Aim* self){
  const auto& c = self->c;
  if (!c.ovl || !c.dz || !c.dz_show) return;
  // 选择绘制中心：有 aim 点用 aim 点；否则用 ctr
  float px = (float)self->ctr.x;
  float py = (float)self->ctr.y;
  if (self->aim_ok_) { px = self->aim_pt_.x; py = self->aim_pt_.y; }
  // 夹在虚拟屏幕内
  int vx,vy,vw,vh; GetVirtualScreen(vx,vy,vw,vh);
  px = MX((float)vx, MN((float)(vx+vw-1), px));
  py = MX((float)vy, MN((float)(vy+vh-1), py));
  float r = (float)c.dz_px;
  if (r <= 0.0f) return;
  OverlayColor col = ARGB((uint8_t)c.dz_a, (uint8_t)c.dz_r, (uint8_t)c.dz_g, (uint8_t)c.dz_b);
  // 形状选择：0=box, 1=circle(多段线), 2=diamond(四边形)
  if (c.dz_shape == 0){
    self->gover.AddRectL({ px - r, py - r, 2*r, 2*r }, col, (float)c.dz_th, (uint16_t)c.dz_layer);
  } else if (c.dz_shape == 2){
    // diamond：四条线（上-右-下-左）
    float x0 = px, y0 = py - r;
    float x1 = px + r, y1 = py;
    float x2 = px, y2 = py + r;
    float x3 = px - r, y3 = py;
    self->gover.AddLineL(x0,y0,x1,y1,(float)c.dz_th,col,(uint16_t)c.dz_layer);
    self->gover.AddLineL(x1,y1,x2,y2,(float)c.dz_th,col,(uint16_t)c.dz_layer);
    self->gover.AddLineL(x2,y2,x3,y3,(float)c.dz_th,col,(uint16_t)c.dz_layer);
    self->gover.AddLineL(x3,y3,x0,y0,(float)c.dz_th,col,(uint16_t)c.dz_layer);
  } else {
    // circle：近似为 N 边形
    int seg = std::max(3, c.dz_segments);
    const float PI = 3.14159265358979323846f;
    float th = (float)c.dz_th;
    uint16_t ly = (uint16_t)c.dz_layer;
    float prevx = px + r, prevy = py;
    for (int i=1;i<=seg;++i){
      float ang = (2.0f*PI*i)/seg;
      float cx = px + r*std::cos(ang);
      float cy = py + r*std::sin(ang);
      self->gover.AddLineL(prevx, prevy, cx, cy, th, col, ly);
      prevx = cx; prevy = cy;
    }
  }
}
