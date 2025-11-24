#include "Game_overlay.h"
#include "Config.hpp"
#include "ovl_impl.hpp"
Game_overlay::Game_overlay() : impl_(new Impl) {}
Game_overlay::~Game_overlay() { Stop(); }

bool Game_overlay::Start() { return impl_->Start(); }
void Game_overlay::Stop() { impl_->Stop(); }
bool Game_overlay::IsRunning() const { return impl_->running.load(); }
void Game_overlay::UseVirtualScreen() { impl_->UseVirtualScreen(); }
void Game_overlay::SetConfig(const Config &cfg) {
  impl_->gpu_enum = cfg.ovl_gpu_enum;
  impl_->gpu_idx = cfg.ovl_gpu_idx;
  impl_->gpu_name = cfg.ovl_gpu_name;
  impl_->gpu_luid = cfg.ovl_gpu_luid;
  impl_->vsync = cfg.ovl_vsync;
  impl_->max_latency = cfg.ovl_max_latency;

  impl_->skip_empty = cfg.ovl_skip_empty;
  impl_->small_window = cfg.ovl_small_window;
  impl_->cfg_win_x = cfg.ovl_win_x;
  impl_->cfg_win_y = cfg.ovl_win_y;
  impl_->cfg_win_w = cfg.ovl_win_w;
  impl_->cfg_win_h = cfg.ovl_win_h;

  // center crosshair config
  impl_->ctr_show = cfg.cross_ctr_show;
  impl_->ctr_size = cfg.cross_ctr_size;
  impl_->ctr_th   = cfg.cross_ctr_th;
  impl_->ctr_col  = ARGB((uint8_t)cfg.cross_ctr_a, (uint8_t)cfg.cross_ctr_r, (uint8_t)cfg.cross_ctr_g, (uint8_t)cfg.cross_ctr_b);
}

void Game_overlay::BeginFrame() { impl_->BeginFrame(); }
void Game_overlay::EndFrame() { impl_->EndFrame(); }
void Game_overlay::Tick() { impl_->RenderOne(); }

void Game_overlay::AddRect(const OverlayRect &r, OverlayColor c, float t){ DrawCmd cmd{}; cmd.type=DrawCmd::Rect; cmd.color=c; cmd.thickness=t; cmd.rect=r; impl_->AddCmd(cmd); }
void Game_overlay::AddRectL(const OverlayRect &r, OverlayColor c, float t, uint16_t layer){ DrawCmd cmd{}; cmd.type=DrawCmd::Rect; cmd.color=c; cmd.thickness=t; cmd.layer=layer; cmd.rect=r; impl_->AddCmd(cmd); }
void Game_overlay::FillRect(const OverlayRect &r, OverlayColor c){ DrawCmd cmd{}; cmd.type=DrawCmd::RectFilled; cmd.color=c; cmd.rect=r; impl_->AddCmd(cmd); }
void Game_overlay::FillRectL(const OverlayRect &r, OverlayColor c, uint16_t layer){ DrawCmd cmd{}; cmd.type=DrawCmd::RectFilled; cmd.color=c; cmd.layer=layer; cmd.rect=r; impl_->AddCmd(cmd); }
void Game_overlay::AddText(float x,float y,const std::wstring &text,float sizePx,OverlayColor c,const std::wstring &font){ DrawCmd cmd{}; cmd.type=DrawCmd::Text; cmd.color=c; cmd.text=text; cmd.textPos={x,y,sizePx}; cmd.fontName=font; impl_->AddCmd(cmd); }
void Game_overlay::AddTextL(float x,float y,const std::wstring &text,float sizePx,OverlayColor c,const std::wstring &font,uint16_t layer){ DrawCmd cmd{}; cmd.type=DrawCmd::Text; cmd.color=c; cmd.text=text; cmd.textPos={x,y,sizePx}; cmd.fontName=font; cmd.layer=layer; impl_->AddCmd(cmd); }
void Game_overlay::AddTextL(float x,float y,const std::wstring &text,float sizePx,OverlayColor c,uint16_t layer){ DrawCmd cmd{}; cmd.type=DrawCmd::Text; cmd.color=c; cmd.text=text; cmd.textPos={x,y,sizePx}; cmd.layer=layer; impl_->AddCmd(cmd); }
void Game_overlay::AddLine(float x1,float y1,float x2,float y2,float th,OverlayColor c){ DrawCmd cmd{}; cmd.type=DrawCmd::Line; cmd.color=c; cmd.thickness=th; cmd.line={x1,y1,x2,y2}; impl_->AddCmd(cmd); }
void Game_overlay::AddLineL(float x1,float y1,float x2,float y2,float th,OverlayColor c,uint16_t layer){ DrawCmd cmd{}; cmd.type=DrawCmd::Line; cmd.color=c; cmd.thickness=th; cmd.layer=layer; cmd.line={x1,y1,x2,y2}; impl_->AddCmd(cmd); }
