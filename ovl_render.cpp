#include "ovl_impl.hpp"
#include "ovl_util.hpp"
#include <d2d1_1.h>
#include <dwrite.h>

void Game_overlay::Impl::BeginFrame()
{
    std::lock_guard<std::mutex> lk(pendingMutex);
    pending.cmds.clear();
}

void Game_overlay::Impl::EndFrame()
{
    auto dl = std::make_shared<DrawList>();
    {
        std::lock_guard<std::mutex> lk(pendingMutex);
        dl->cmds = pending.cmds;
    }
    std::atomic_store_explicit(&current, dl, std::memory_order_release);
}

void Game_overlay::Impl::AddCmd(const DrawCmd& c)
{
    std::lock_guard<std::mutex> lk(pendingMutex);
    pending.cmds.push_back(c);
}

HRESULT Game_overlay::Impl::CreateTextFormat(
    const std::wstring& font, float size,
    IDWriteTextFormat** out)
{
    uint64_t key =
        (std::hash<std::wstring>{}(font) ^ (uint64_t(std::lround(size * 100)) << 1));

    auto it = textFormatCache.find(key);
    if (it != textFormatCache.end())
    {
        *out = it->second.Get();
        (*out)->AddRef();
        return S_OK;
    }

    ComPtr<IDWriteTextFormat> fmt;
    if (FAILED(dwFactory->CreateTextFormat(
        font.c_str(), nullptr,
        DWRITE_FONT_WEIGHT_NORMAL,
        DWRITE_FONT_STYLE_NORMAL,
        DWRITE_FONT_STRETCH_NORMAL,
        size, L"en-US", &fmt)))
        return E_FAIL;

    fmt->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP);
    textFormatCache.emplace(key, fmt);
    *out = fmt.Detach();
    return S_OK;
}

HRESULT Game_overlay::Impl::CreateTargets()
{
    ReleaseTargets();

    ComPtr<ID3D11Texture2D> bb;
    if (FAILED(swapChain->GetBuffer(0, IID_PPV_ARGS(&bb)))) { lg::werr(P_E, "GetBuffer"); return E_FAIL; }

    if (FAILED(d3d->CreateRenderTargetView(bb.Get(), nullptr, &rtv))) { lg::werr(P_E, "CreateRenderTargetView"); return E_FAIL; }

    if (!d2dFactory)
    {
        D2D1_FACTORY_OPTIONS opts{};
#if defined(_DEBUG)
        opts.debugLevel = D2D1_DEBUG_LEVEL_INFORMATION;
#endif
        if (FAILED(D2D1CreateFactory(
            D2D1_FACTORY_TYPE_MULTI_THREADED,
            __uuidof(ID2D1Factory1),
            &opts,
            reinterpret_cast<void**>(d2dFactory.GetAddressOf())))) {
            lg::werr(P_E, "D2D1CreateFactory");
            return E_FAIL;
        }
    }

    if (!d2dDevice)
    {
        ComPtr<IDXGIDevice> dxgiDev; d3d.As(&dxgiDev);
        if (FAILED(d2dFactory->CreateDevice(dxgiDev.Get(), &d2dDevice))) { lg::werr(P_E, "CreateDevice(D2D) "); return E_FAIL; }
    }

    if (!d2dCtx)
    {
        if (FAILED(d2dDevice->CreateDeviceContext(
            D2D1_DEVICE_CONTEXT_OPTIONS_NONE, &d2dCtx))) { lg::werr(P_E, "CreateDeviceContext"); return E_FAIL; }
    }

    d2dCtx->SetUnitMode(D2D1_UNIT_MODE_PIXELS);
    d2dCtx->SetDpi(96.f, 96.f);

    ComPtr<IDXGISurface> surf;
    if (FAILED(bb.As(&surf))) { lg::werr(P_E, "As(IdxgiSurface)"); return E_FAIL; }

    D2D1_BITMAP_PROPERTIES1 props =
        D2D1::BitmapProperties1(
            D2D1_BITMAP_OPTIONS_TARGET |
            D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
            D2D1::PixelFormat(
                DXGI_FORMAT_B8G8R8A8_UNORM,
                D2D1_ALPHA_MODE_PREMULTIPLIED),
            96.f, 96.f
        );

    if (FAILED(d2dCtx->CreateBitmapFromDxgiSurface(
        surf.Get(), &props, &d2dTarget))) { lg::werr(P_E, "CreateBitmapFromDxgiSurface"); return E_FAIL; }

    d2dCtx->SetTarget(d2dTarget.Get());

    return S_OK;
}

void Game_overlay::Impl::ReleaseTargets()
{
    if (d2dCtx) d2dCtx->SetTarget(nullptr);
    d2dTarget.Reset();
    rtv.Reset();
}

void Game_overlay::Impl::RenderOne()
{
    if (!d2dCtx || !swapChain) return;

   auto dl = std::atomic_load_explicit(&current, std::memory_order_acquire);
   bool wantShow = visible.load() && (!skip_empty || (dl && !dl->cmds.empty()));
   ShowWindow(hwnd, wantShow ? SW_SHOWNA : SW_HIDE);
   if (!wantShow) return;

    d2dCtx->BeginDraw();

    d2dCtx->Clear(D2D1::ColorF(0.f, 0.f, 0.f, 0.f));

    D2D1_MATRIX_3X2_F __oldM;
    if (small_window) {
        d2dCtx->GetTransform(&__oldM);
        d2dCtx->SetTransform(D2D1::Matrix3x2F::Translation((float)-winX, (float)-winY));
    }

    if (dl)
    {
        for (const auto& c : dl->cmds)
        {
            switch (c.type)
            {
            case DrawCmd::Rect:
                brush->SetColor(ToD2DColor(c.color));
                d2dCtx->DrawRectangle(
                    D2D1::RectF(c.rect.x, c.rect.y,
                        c.rect.x + c.rect.w, c.rect.y + c.rect.h),
                    brush.Get(), c.thickness);
                break;

            case DrawCmd::RectFilled:
                brush->SetColor(ToD2DColor(c.color));
                d2dCtx->FillRectangle(
                    D2D1::RectF(c.rect.x, c.rect.y,
                        c.rect.x + c.rect.w, c.rect.y + c.rect.h),
                    brush.Get());
                break;

            case DrawCmd::Text:
            {
                brush->SetColor(ToD2DColor(c.color));
                ComPtr<IDWriteTextFormat> fmt;
                if (SUCCEEDED(CreateTextFormat(
                    c.fontName.empty() ? L"Segoe UI" : c.fontName,
                    c.textPos.size, &fmt)))
                {
                    D2D1_RECT_F layout = D2D1::RectF(
                        c.textPos.x, c.textPos.y,
                        c.textPos.x + 4000.f, c.textPos.y + 1200.f);
                    d2dCtx->DrawTextW(
                        c.text.c_str(),
                        (UINT32)c.text.size(),
                        fmt.Get(),
                        layout,
                        brush.Get());
                }
            } break;

            }
        }
    }

    if (small_window) {
        d2dCtx->SetTransform(__oldM);
    }

    HRESULT hrEnd = d2dCtx->EndDraw();
    if (FAILED(hrEnd))
    {
        lg::hrfail(P_E, "EndDraw", hrEnd);
        CreateTargets();
    }

    HRESULT phr = S_OK;
    if (vsync) {
        phr = swapChain->Present(1, 0);
    } else {
        phr = swapChain->Present(0, 0);
    }
    if (phr == DXGI_STATUS_OCCLUDED || phr == DXGI_ERROR_ACCESS_LOST) {
        return;
    }
}
