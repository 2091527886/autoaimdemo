#include "Game_overlay.h"
#include "Log.hpp"
#include "Config.hpp"
#include "SystemTuner.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dwmapi.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <dxgi1_3.h>
#include <d2d1_1.h>
#include <dwrite.h>
#include <dcomp.h>
#include <wrl/client.h>

#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <cwctype>

#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "dcomp.lib")

using Microsoft::WRL::ComPtr;

static void GetVirtualScreen(int& x, int& y, int& w, int& h)
{
    x = GetSystemMetrics(SM_XVIRTUALSCREEN);
    y = GetSystemMetrics(SM_YVIRTUALSCREEN);
    w = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    h = GetSystemMetrics(SM_CYVIRTUALSCREEN);
}

static void SetPerMonitorV2DpiAwareness()
{
    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    if (!user32) return;
    using Fn = BOOL(WINAPI*)(HANDLE);
    if (auto p = reinterpret_cast<Fn>(GetProcAddress(user32, "SetProcessDpiAwarenessContext")))
        p(reinterpret_cast<HANDLE>(-4)); // PER_MONITOR_AWARE_V2
}

static D2D1_COLOR_F ToD2DColor(OverlayColor argb)
{
    float a = float((argb >> 24) & 0xFF) / 255.f;
    float r = float((argb >> 16) & 0xFF) / 255.f;
    float g = float((argb >> 8) & 0xFF) / 255.f;
    float b = float((argb >> 0) & 0xFF) / 255.f;
    return D2D1::ColorF(r, g, b, a);
}

// GPU helpers (overlay-local)
static std::wstring wenv(const wchar_t* k)
{
    wchar_t buf[512];
    DWORD n = GetEnvironmentVariableW(k, buf, (DWORD)_countof(buf));
    if (n == 0 || n >= _countof(buf)) return L"";
    return std::wstring(buf, buf + n);
}

static std::wstring wlower(std::wstring s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](wchar_t c){ return std::towlower(c); });
    return s;
}

static bool parse_idx(const std::wstring& s, int& out)
{
    if (s.empty()) return false;
    try { out = std::stoi(s); return out >= 0; } catch (...) { return false; }
}

static bool parse_luid(const std::wstring& s, LUID& out)
{
    // format: hi:lo (hex or dec)
    size_t p = s.find(L':');
    if (p == std::wstring::npos) return false;
    auto sh = s.substr(0, p), sl = s.substr(p + 1);
    auto to_u32 = [](const std::wstring& t, DWORD& v)->bool {
        try {
            size_t idx = 0; int base = 10;
            if (t.size() > 2 && (t.rfind(L"0x",0)==0 || t.rfind(L"0X",0)==0)) { base = 16; }
            v = (DWORD)std::stoul(t, &idx, base);
            return idx == t.size();
        } catch (...) { return false; }
    };
    DWORD hi=0, lo=0;
    if (!to_u32(sh, hi) || !to_u32(sl, lo)) return false;
    out.HighPart = (LONG)hi;
    out.LowPart  = (DWORD)lo;
    return true;
}

struct GpuInfo { ComPtr<IDXGIAdapter1> adp; DXGI_ADAPTER_DESC1 desc{}; int idx=-1; };

static std::vector<GpuInfo> list_adapters()
{
    std::vector<GpuInfo> v;
    ComPtr<IDXGIFactory1> fac;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&fac)))) return v;
    for (UINT i=0;;++i) {
        ComPtr<IDXGIAdapter1> a; DXGI_ADAPTER_DESC1 d{};
        if (fac->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
        if (!a) break;
        if (FAILED(a->GetDesc1(&d))) continue;
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue; // skip WARP
        v.push_back({ a, d, (int)i });
    }
    return v;
}

static void log_adapters(const std::vector<GpuInfo>& v)
{
    for (auto& g: v) {
        std::wstring name(g.desc.Description);
        double mb = (double)g.desc.DedicatedVideoMemory / (1024.0*1024.0);
        lg::o() << P_I "OVL GPU[" << g.idx << "] "
                << std::string(name.begin(), name.end())
                << " | ven=0x" << std::hex << std::uppercase << g.desc.VendorId
                << " dev=0x" << g.desc.DeviceId << std::dec << std::nouppercase
                << " | memMB=" << (int)mb
                << " | luid=" << g.desc.AdapterLuid.HighPart << ":" << g.desc.AdapterLuid.LowPart
                << lg::endl;
    }
}

// pick by Config.SetConfig
static ComPtr<IDXGIAdapter1> pick_adapter_cfg(const std::wstring& name,
                                              const std::wstring& luid,
                                              int idx,
                                              bool do_enum)
{
    auto v = list_adapters();
    if (do_enum) log_adapters(v);
    if (v.empty()) return {};
    if (!name.empty()) {
        auto kn = wlower(name);
        for (auto& g: v) {
            std::wstring dn(g.desc.Description);
            if (wlower(dn).find(kn) != std::wstring::npos) return g.adp;
        }
    }
    if (!luid.empty()) {
        LUID t{}; if (parse_luid(luid, t)) {
            for (auto& g: v) {
                if (g.desc.AdapterLuid.HighPart==t.HighPart && g.desc.AdapterLuid.LowPart==t.LowPart)
                    return g.adp;
            }
        }
    }
    if (idx >= 0) {
        for (auto& g: v) if (g.idx==idx) return g.adp;
    }
    return {};
}

struct DrawCmd
{
    enum Type { Rect, RectFilled, Text } type;
    OverlayColor color = 0;
    float thickness = 1.0f;
    union
    {
        OverlayRect rect;
        struct { float x, y, size; } textPos;
    };
    std::wstring text;
    std::wstring fontName;
};

struct DrawList
{
    std::vector<DrawCmd> cmds;
};

struct Game_overlay::Impl
{
    HINSTANCE hinst = nullptr;
    HWND hwnd = nullptr;
    int winX = 0, winY = 0, winW = 0, winH = 0;
    bool useVirtual = true;

    ComPtr<ID3D11Device>           d3d;
    ComPtr<ID3D11DeviceContext>    d3dCtx;
    ComPtr<IDXGISwapChain1>        swapChain;
    ComPtr<ID3D11RenderTargetView> rtv;

    ComPtr<ID2D1Factory1>          d2dFactory;
    ComPtr<ID2D1Device>            d2dDevice;
    ComPtr<ID2D1DeviceContext>     d2dCtx;
    ComPtr<ID2D1Bitmap1>           d2dTarget;
    ComPtr<ID2D1SolidColorBrush>   brush;

    ComPtr<IDWriteFactory>         dwFactory;
    std::unordered_map<uint64_t, ComPtr<IDWriteTextFormat>> textFormatCache;

    ComPtr<IDCompositionDevice>    dcompDevice;
    ComPtr<IDCompositionTarget>    dcompTarget;
    ComPtr<IDCompositionVisual>    dcompRoot;

    std::thread thread;
    std::atomic<bool> running{ false };
    std::atomic<bool> visible{ true };
    std::atomic<unsigned> maxFps{ 0 };

    std::mutex pendingMutex;
    DrawList pending;
    std::shared_ptr<DrawList> current;

    // config view (overlay-only fields used)
    std::wstring gpu_name;
    std::wstring gpu_luid;
    int gpu_idx = -1;
    bool gpu_enum = false;
    bool vsync = false; // 默认不使用 vsync
    int  max_latency = 1;

    bool skip_empty = true;
    int  idle_sleep_ms = 50;

    // small window
    bool small_window = true;
    int  cfg_win_x = 0, cfg_win_y = 0, cfg_win_w = 640, cfg_win_h = 640;

    void UseVirtualScreen() { useVirtual = true; }
    void SetBounds(int x, int y, int w, int h) { useVirtual = false; winX = x; winY = y; winW = w; winH = h; }

    bool Start();
    void Stop();

    void BeginFrame();
    void EndFrame();
    void AddCmd(const DrawCmd&);

    static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
    bool    CreateWindowAndDevices();
    void    DestroyWindowAndDevices();
    void    RenderLoop();
    void    RenderOne();

    HRESULT CreateTextFormat(const std::wstring& font, float size, IDWriteTextFormat** out);
    HRESULT CreateTargets();
    void    ReleaseTargets();
};

Game_overlay::Game_overlay() : impl_(new Impl) {}
Game_overlay::~Game_overlay() { Stop(); }

bool Game_overlay::Start() { return impl_->Start(); }
void Game_overlay::Stop() { impl_->Stop(); }
bool Game_overlay::IsRunning() const { return impl_->running.load(); }
void Game_overlay::UseVirtualScreen() { impl_->UseVirtualScreen(); }
void Game_overlay::SetMaxFPS(unsigned f) { impl_->maxFps.store(f); }

void Game_overlay::SetConfig(const Config& cfg)
{
    // copy subset for overlay gpu/vsync/latency
    impl_->gpu_enum    = cfg.ovl_gpu_enum;
    impl_->gpu_idx     = cfg.ovl_gpu_idx;
    impl_->gpu_name    = cfg.ovl_gpu_name;
    impl_->gpu_luid    = cfg.ovl_gpu_luid;
    impl_->vsync       = cfg.ovl_vsync;
    impl_->max_latency = cfg.ovl_max_latency;

    impl_->skip_empty     = cfg.ovl_skip_empty;
    impl_->idle_sleep_ms  = cfg.ovl_idle_sleep_ms;
    impl_->small_window   = cfg.ovl_small_window;
    impl_->cfg_win_x      = cfg.ovl_win_x;
    impl_->cfg_win_y      = cfg.ovl_win_y;
    impl_->cfg_win_w      = cfg.ovl_win_w;
    impl_->cfg_win_h      = cfg.ovl_win_h;
}

void Game_overlay::BeginFrame() { impl_->BeginFrame(); }
void Game_overlay::EndFrame() { impl_->EndFrame(); }

void Game_overlay::AddRect(const OverlayRect& r, OverlayColor c, float t)
{
    DrawCmd cmd{}; cmd.type = DrawCmd::Rect; cmd.color = c; cmd.thickness = t; cmd.rect = r;
    impl_->AddCmd(cmd);
}
void Game_overlay::FillRect(const OverlayRect& r, OverlayColor c)
{
    DrawCmd cmd{}; cmd.type = DrawCmd::RectFilled; cmd.color = c; cmd.rect = r;
    impl_->AddCmd(cmd);
}
void Game_overlay::AddText(float x, float y, const std::wstring& text,
    float sizePx, OverlayColor c, const std::wstring& font)
{
    DrawCmd cmd{}; cmd.type = DrawCmd::Text; cmd.color = c; cmd.text = text;
    cmd.textPos = { x, y, sizePx }; cmd.fontName = font;
    impl_->AddCmd(cmd);
}
bool Game_overlay::Impl::Start()
{
    if (running.load()) return true;
    running = true;
    thread = std::thread([this] {
        if (this->running.load()) {
            // Optional: overlay-specific affinity
            // affinity handled in OvlAsync/OverlayAnim
        }
        SetPerMonitorV2DpiAwareness();
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        hinst = GetModuleHandleW(nullptr);
        if (!CreateWindowAndDevices()) {
            running = false;
            CoUninitialize();
            return;
        }
        RenderLoop();
        DestroyWindowAndDevices();
        CoUninitialize();
        });
    return true;
}

void Game_overlay::Impl::Stop()
{
    if (!running.exchange(false)) return;
    if (hwnd) PostMessageW(hwnd, WM_CLOSE, 0, 0);
    if (thread.joinable()) thread.join();
}

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

LRESULT CALLBACK Game_overlay::Impl::WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    Impl* self = reinterpret_cast<Impl*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
    switch (msg)
    {
    case WM_NCCREATE:
    {
        auto cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
        SetWindowLongPtrW(hWnd, GWLP_USERDATA,
            reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
        return TRUE;
    }
    case WM_NCHITTEST:     return HTTRANSPARENT;
    case WM_MOUSEACTIVATE: return MA_NOACTIVATE;
    case WM_SIZE:
        if (self && self->swapChain)
        {
            UINT w = LOWORD(lParam), h = HIWORD(lParam);
            if (w > 0 && h > 0)
            {
                self->ReleaseTargets();
                self->swapChain->ResizeBuffers(0, w, h, DXGI_FORMAT_UNKNOWN, 0);
                self->CreateTargets();
                if (self->dcompDevice) self->dcompDevice->Commit();
            }
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}

bool Game_overlay::Impl::CreateWindowAndDevices()
{
    if (small_window) {
        useVirtual = false;
        int vx, vy, vw, vh; GetVirtualScreen(vx, vy, vw, vh);
        winW = (cfg_win_w > 0) ? cfg_win_w : 640;
        winH = (cfg_win_h > 0) ? cfg_win_h : 640;
        int cx = vx + (vw - winW) / 2;
        int cy = vy + (vh - winH) / 2;
        winX = (cfg_win_x != 0) ? cfg_win_x : cx;
        winY = (cfg_win_y != 0) ? cfg_win_y : cy;
    }
    else if (useVirtual) GetVirtualScreen(winX, winY, winW, winH);
    else if (winW <= 0 || winH <= 0) GetVirtualScreen(winX, winY, winW, winH);

    WNDCLASSEXW wc{ sizeof(WNDCLASSEXW) };
    wc.lpfnWndProc = Game_overlay::Impl::WndProc;
    wc.hInstance = hinst;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.lpszClassName = L"GameOverlayDCompWnd";
    wc.hbrBackground = nullptr;
    if (!RegisterClassExW(&wc)) { lg::werr(P_E, "RegisterClassExW"); return false; }

    DWORD ex = WS_EX_TOPMOST | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_LAYERED;
    hwnd = CreateWindowExW(ex, wc.lpszClassName, L"", WS_POPUP,
        winX, winY, winW, winH, nullptr, nullptr, hinst, this);
    if (!hwnd) { lg::werr(P_E, "CreateWindowExW"); return false; }

    BOOL dwm = FALSE;
    if (SUCCEEDED(DwmIsCompositionEnabled(&dwm)) && dwm)
    {
        MARGINS m = { -1, -1, -1, -1 };
        DwmExtendFrameIntoClientArea(hwnd, &m);
    }

    ShowWindow(hwnd, SW_SHOWNA);
    SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE);
    SetWindowPos(hwnd, HWND_TOPMOST, winX, winY, winW, winH,
        SWP_NOACTIVATE | SWP_SHOWWINDOW);

    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#if defined(_DEBUG)
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0
    };
    D3D_FEATURE_LEVEL flOut{};
    HRESULT hr = E_FAIL;
    {
        // try config-selected adapter first
        ComPtr<IDXGIAdapter1> pick = pick_adapter_cfg(gpu_name, gpu_luid, gpu_idx, gpu_enum);
        if (pick) {
            hr = D3D11CreateDevice(
                pick.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags,
                featureLevels, _countof(featureLevels),
                D3D11_SDK_VERSION, &d3d, &flOut, &d3dCtx);
            if (FAILED(hr)) {
                lg::hrfail(P_E, "D3D11CreateDevice(pick)", hr);
            } else {
                DXGI_ADAPTER_DESC1 d{}; pick->GetDesc1(&d);
                std::wstring n(d.Description);
                lg::o() << P_I "OVL use GPU: " << ([](const std::wstring& ws){
                    int len = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), nullptr, 0, nullptr, nullptr);
                    std::string out; out.resize(std::max(0, len));
                    if (len>0) WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), out.data(), len, nullptr, nullptr);
                    return out; })(n) << lg::endl;
            }
        }
        if (FAILED(hr)) {
            // fallback to default
            hr = D3D11CreateDevice(
                nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
                featureLevels, _countof(featureLevels),
                D3D11_SDK_VERSION, &d3d, &flOut, &d3dCtx);
        }
    }
    if (FAILED(hr))
    {
        lg::hrfail(P_E, "D3D11CreateDevice", hr);
        return false;
    }

    // DXGI factory
    ComPtr<IDXGIDevice> dxgiDev; d3d.As(&dxgiDev);
    ComPtr<IDXGIAdapter> adapter; dxgiDev->GetAdapter(&adapter);
    ComPtr<IDXGIFactory2> factory2;
    {
        ComPtr<IDXGIFactory> baseFactory;
        adapter->GetParent(IID_PPV_ARGS(&baseFactory));
        baseFactory.As(&factory2);
    }
    if (!factory2) return false;

    DXGI_SWAP_CHAIN_DESC1 scd{};
    scd.Width = winW;
    scd.Height = winH;
    scd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    scd.SampleDesc.Count = 1;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.BufferCount = 3; // triple buffering for smoother
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    scd.AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED;
    scd.Scaling = DXGI_SCALING_STRETCH;
    scd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

    if (FAILED(factory2->CreateSwapChainForComposition(
        d3d.Get(), &scd, nullptr, &swapChain)))
    {
        lg::werr(P_E, "CreateSwapChainForComposition");
        return false;
    }

    // set maximum frame latency if configured (>0)
    ComPtr<IDXGIDevice1> dxgiDev1; d3d.As(&dxgiDev1);
    if (dxgiDev1 && max_latency > 0) dxgiDev1->SetMaximumFrameLatency((UINT)std::min(std::max(1, max_latency), 3));

    if (FAILED(DCompositionCreateDevice(dxgiDev.Get(), IID_PPV_ARGS(&dcompDevice)))) { lg::werr(P_E, "DCompositionCreateDevice"); return false; }
    if (FAILED(dcompDevice->CreateTargetForHwnd(hwnd, TRUE, &dcompTarget))) { lg::werr(P_E, "CreateTargetForHwnd"); return false; }
    if (FAILED(dcompDevice->CreateVisual(&dcompRoot))) { lg::werr(P_E, "CreateVisual"); return false; }
    if (FAILED(dcompRoot->SetContent(swapChain.Get()))) { lg::werr(P_E, "SetContent"); return false; }
    if (FAILED(dcompTarget->SetRoot(dcompRoot.Get()))) { lg::werr(P_E, "SetRoot"); return false; }
    dcompDevice->Commit();

    if (FAILED(CreateTargets())) { lg::werr(P_E, "CreateTargets"); return false; }

    if (FAILED(d2dCtx->CreateSolidColorBrush(
        D2D1::ColorF(1.f, 1.f, 1.f, 1.f), &brush))) { lg::werr(P_E, "CreateSolidColorBrush"); return false; }
    d2dCtx->SetAntialiasMode(D2D1_ANTIALIAS_MODE_PER_PRIMITIVE);
    d2dCtx->SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_CLEARTYPE);

    if (FAILED(DWriteCreateFactory(
        DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory),
        reinterpret_cast<IUnknown**>(dwFactory.GetAddressOf()))))
    {
        lg::werr(P_E, "DWriteCreateFactory");
        return false;
    }

    std::atomic_store_explicit(
        &current, std::make_shared<DrawList>(),
        std::memory_order_release);

    return true;
}

void Game_overlay::Impl::DestroyWindowAndDevices()
{
    lg::o() << P_I "Overlay.Destroy" << lg::endl;
    ReleaseTargets();
    brush.Reset();
    d2dCtx.Reset();
    d2dDevice.Reset();
    d2dFactory.Reset();
    rtv.Reset();
    swapChain.Reset();
    dcompRoot.Reset();
    dcompTarget.Reset();
    dcompDevice.Reset();
    dwFactory.Reset();
    
    d3dCtx.Reset();
    d3d.Reset();

    if (hwnd) { DestroyWindow(hwnd); hwnd = nullptr; }
    UnregisterClassW(L"GameOverlayDCompWnd", hinst);
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

void Game_overlay::Impl::RenderLoop()
{
    running = true;

    HANDLE frameWait = nullptr;
    ComPtr<IDXGISwapChain2> sc2; if (swapChain) swapChain.As(&sc2);
    if (sc2) frameWait = sc2->GetFrameLatencyWaitableObject();

    MSG msg{};
    while (running.load())
    {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
            {
                running = false;
                break;
            }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (!running.load()) break;

        unsigned cap = maxFps.load();
        DWORD timeout = (cap > 0) ? (1000u / cap) : INFINITE;
        if (frameWait)
        {
            WaitForSingleObject(frameWait, timeout);
        }
        else if (cap > 0)
        {
            Sleep(timeout);
        }

        RenderOne();
    }
}

void Game_overlay::Impl::RenderOne()
{
    if (!d2dCtx || !swapChain) return;

    if (!visible.load())
    {
        ShowWindow(hwnd, SW_HIDE);
        Sleep(10);
        return;
    }
    else
    {
        ShowWindow(hwnd, SW_SHOWNA);
    }

    auto dl = std::atomic_load_explicit(&current, std::memory_order_acquire);

    // 空帧跳过：没有绘制命令时不渲染不Present
    if (skip_empty && (!dl || dl->cmds.empty()))
    {
        ShowWindow(hwnd, SW_HIDE);
        ComPtr<IDXGISwapChain2> sc2; swapChain.As(&sc2);
        HANDLE h = sc2 ? sc2->GetFrameLatencyWaitableObject() : nullptr;
        DWORD ms = (DWORD)std::max(1, idle_sleep_ms);
        if (h) WaitForSingleObject(h, ms); else Sleep(ms);
        return;
    }
    else
    {
        ShowWindow(hwnd, SW_SHOWNA);
    }

    d2dCtx->BeginDraw();

    d2dCtx->Clear(D2D1::ColorF(0.f, 0.f, 0.f, 0.f));

    // 将全屏坐标平移到小窗口局部坐标
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

    // 恢复变换
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
        Sleep(100);
    }
}
