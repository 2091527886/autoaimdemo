#include "ovl_impl.hpp"
HRESULT Game_overlay::Impl::CreateTargets(){
  ReleaseTargets();
  ComPtr<ID3D11Texture2D> bb; if(FAILED(swapChain->GetBuffer(0, IID_PPV_ARGS(&bb)))){ lg::werr(P_E, "GetBuffer"); return E_FAIL; }
  if(FAILED(d3d->CreateRenderTargetView(bb.Get(), nullptr, &rtv))){ lg::werr(P_E, "CreateRenderTargetView"); return E_FAIL; }
  if(!d2dFactory){ D2D1_FACTORY_OPTIONS opts{}; 
#if defined(_DEBUG)
    opts.debugLevel=D2D1_DEBUG_LEVEL_INFORMATION;
#endif
    if(FAILED(D2D1CreateFactory(D2D1_FACTORY_TYPE_MULTI_THREADED, __uuidof(ID2D1Factory1), &opts, reinterpret_cast<void**>(d2dFactory.GetAddressOf())))){
      lg::werr(P_E, "D2D1CreateFactory"); return E_FAIL; }
  }
  if(!d2dDevice){ ComPtr<IDXGIDevice> dxgiDev; d3d.As(&dxgiDev); if(FAILED(d2dFactory->CreateDevice(dxgiDev.Get(), &d2dDevice))){ lg::werr(P_E, "CreateDevice(D2D) "); return E_FAIL; } }
  if(!d2dCtx){ if(FAILED(d2dDevice->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE, &d2dCtx))){ lg::werr(P_E, "CreateDeviceContext"); return E_FAIL; } }
  d2dCtx->SetUnitMode(D2D1_UNIT_MODE_PIXELS); d2dCtx->SetDpi(96.f,96.f);
  ComPtr<IDXGISurface> surf; if(FAILED(bb.As(&surf))){ lg::werr(P_E, "As(IdxgiSurface)"); return E_FAIL; }
  D2D1_BITMAP_PROPERTIES1 props = D2D1::BitmapProperties1(D2D1_BITMAP_OPTIONS_TARGET|D2D1_BITMAP_OPTIONS_CANNOT_DRAW, D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED), 96.f, 96.f);
  if(FAILED(d2dCtx->CreateBitmapFromDxgiSurface(surf.Get(), &props, &d2dTarget))){ lg::werr(P_E, "CreateBitmapFromDxgiSurface"); return E_FAIL; }
  d2dCtx->SetTarget(d2dTarget.Get());
  return S_OK;
}

void Game_overlay::Impl::ReleaseTargets(){ if(d2dCtx) d2dCtx->SetTarget(nullptr); d2dTarget.Reset(); rtv.Reset(); }
