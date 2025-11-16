#include "ovl_impl.hpp"
#include "ovl_util.hpp"
#include "pch.hpp"
void Game_overlay::Impl::RenderOne(){
  if(!d2dCtx || !swapChain) return;
  auto dl = std::atomic_load_explicit(&current, std::memory_order_acquire);
  bool wantShow = visible.load() && (!skip_empty || (dl && !dl->cmds.empty()));
  ShowWindow(hwnd, wantShow ? SW_SHOWNA : SW_HIDE);
  if(!wantShow) return;
  d2dCtx->BeginDraw();
  d2dCtx->Clear(D2D1::ColorF(0.f,0.f,0.f,0.f));
  D2D1_MATRIX_3X2_F __oldM; if(small_window){ d2dCtx->GetTransform(&__oldM); d2dCtx->SetTransform(D2D1::Matrix3x2F::Translation((float)-winX,(float)-winY)); }
  if(dl){ for(const auto& c : dl->cmds){ switch(c.type){
    case DrawCmd::Rect: brush->SetColor(C(c.color)); d2dCtx->DrawRectangle(RF(c.rect.x, c.rect.y, c.rect.w, c.rect.h), brush.Get(), c.thickness); break;
    case DrawCmd::RectFilled: brush->SetColor(C(c.color)); d2dCtx->FillRectangle(RF(c.rect.x, c.rect.y, c.rect.w, c.rect.h), brush.Get()); break;
    case DrawCmd::Text: { brush->SetColor(C(c.color)); OVL_DRAW_TEXT(d2dCtx, c.text, c.textPos.x, c.textPos.y, c.textPos.size, c.color, c.fontName); } break; 
    case DrawCmd::Line: brush->SetColor(C(c.color)); d2dCtx->DrawLine(D2D1::Point2F(c.line.x1, c.line.y1), D2D1::Point2F(c.line.x2, c.line.y2), brush.Get(), c.thickness); break; }
    }
  }
  if(small_window){ d2dCtx->SetTransform(__oldM); }
  HRESULT hrEnd = d2dCtx->EndDraw(); if(FAILED(hrEnd)){ lg::hrfail(P_E, "EndDraw", hrEnd); CreateTargets(); }
  HRESULT phr=S_OK; phr = vsync ? swapChain->Present(1,0) : swapChain->Present(0,0);
  if(phr == DXGI_STATUS_OCCLUDED || phr == DXGI_ERROR_ACCESS_LOST) return;
}
