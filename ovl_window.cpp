#include "Config.hpp"
#include "ovl_gpu.hpp"
#include "ovl_impl.hpp"
#include "ovl_util.hpp"
#include "pch.hpp"

bool Game_overlay::Impl::Start() {
  if (running.load())
    return true;
  running = true;
  SetPerMonitorV2DpiAwareness();
  CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  hinst = GetModuleHandleW(nullptr);
  if (!CreateWindowAndDevices()) {
    running = false;
    CoUninitialize();
    return false;
  }
  return true;
}

void Game_overlay::Impl::Stop() {
  if (!running.exchange(false))
    return;
  if (hwnd)
    PostMessageW(hwnd, WM_CLOSE, 0, 0);
  DestroyWindowAndDevices();
  CoUninitialize();
}

LRESULT CALLBACK Game_overlay::Impl::WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  Impl *self = reinterpret_cast<Impl *>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
  switch (msg) {
  case WM_NCCREATE: {
    auto cs = reinterpret_cast<CREATESTRUCTW *>(lParam);
    SetWindowLongPtrW(hWnd, GWLP_USERDATA,
                      reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
    return TRUE;
  }
  case WM_NCHITTEST:
    return HTTRANSPARENT;
  case WM_MOUSEACTIVATE:
    return MA_NOACTIVATE;
  case WM_SIZE:
    if (self && self->swapChain) {
      UINT w = LOWORD(lParam), h = HIWORD(lParam);
      if (w > 0 && h > 0) {
        self->ReleaseTargets();
        self->swapChain->ResizeBuffers(0, w, h, DXGI_FORMAT_UNKNOWN, 0);
        self->CreateTargets();
        if (self->dcompDevice)
          self->dcompDevice->Commit();
      }
    }
    break;
  case WM_DESTROY:
    PostQuitMessage(0);
    break;
  }
  return DefWindowProcW(hWnd, msg, wParam, lParam);
}

bool Game_overlay::Impl::CreateWindowAndDevices() {
  if (small_window) {
    useVirtual = false;
    int vx, vy, vw, vh;
    GetVirtualScreen(vx, vy, vw, vh);
    winW = (cfg_win_w > 0) ? cfg_win_w : 640;
    winH = (cfg_win_h > 0) ? cfg_win_h : 640;
    int cx = vx + (vw - winW) / 2, cy = vy + (vh - winH) / 2;
    winX = (cfg_win_x != 0) ? cfg_win_x : cx;
    winY = (cfg_win_y != 0) ? cfg_win_y : cy;
  } else if (useVirtual)
    GetVirtualScreen(winX, winY, winW, winH);
  else if (winW <= 0 || winH <= 0)
    GetVirtualScreen(winX, winY, winW, winH);

  WNDCLASSEXW wc{sizeof(WNDCLASSEXW)};
  wc.lpfnWndProc = Game_overlay::Impl::WndProc;
  wc.hInstance = hinst;
  wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
  wc.lpszClassName = L"GameOverlayDCompWnd";
  wc.hbrBackground = nullptr;
  if (!RegisterClassExW(&wc)) {
    lg::werr(P_E, "RegisterClassExW");
    return false;
  }

  DWORD ex = WS_EX_TOPMOST | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_LAYERED;
  hwnd = CreateWindowExW(ex, wc.lpszClassName, L"", WS_POPUP,
                         winX, winY, winW, winH, nullptr, nullptr, hinst, this);
  if (!hwnd) {
    lg::werr(P_E, "CreateWindowExW");
    return false;
  }

  BOOL dwm = FALSE;
  if (SUCCEEDED(DwmIsCompositionEnabled(&dwm)) && dwm) {
    MARGINS m = {-1, -1, -1, -1};
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
      D3D_FEATURE_LEVEL_10_0};
  D3D_FEATURE_LEVEL flOut{};
  HRESULT hr = E_FAIL;
  {
    ComPtr<IDXGIAdapter1> pick = pick_adapter_cfg(gpu_name, gpu_luid, gpu_idx, gpu_enum);
    if (pick) {
      hr = D3D11CreateDevice(
          pick.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags,
          featureLevels, _countof(featureLevels),
          D3D11_SDK_VERSION, &d3d, &flOut, &d3dCtx);
      if (FAILED(hr)) {
        lg::hrfail(P_E, "D3D11CreateDevice(pick)", hr);
      } else {
        DXGI_ADAPTER_DESC1 d{};
        pick->GetDesc1(&d);
        std::wstring n(d.Description);
        lg::o() << P_I "OVL use GPU: " << ([](const std::wstring &ws) {
                    int len = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), nullptr, 0, nullptr, nullptr);
                    std::string out; out.resize(std::max(0, len));
                    if (len>0) WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), out.data(), len, nullptr, nullptr);
                    return out; })(n) << lg::endl;
      }
    }
    if (FAILED(hr)) {
      hr = D3D11CreateDevice(
          nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
          featureLevels, _countof(featureLevels),
          D3D11_SDK_VERSION, &d3d, &flOut, &d3dCtx);
    }
  }
  if (FAILED(hr)) {
    lg::hrfail(P_E, "D3D11CreateDevice", hr);
    return false;
  }

  ComPtr<IDXGIDevice> dxgiDev;
  d3d.As(&dxgiDev);
  ComPtr<IDXGIAdapter> adapter;
  dxgiDev->GetAdapter(&adapter);
  ComPtr<IDXGIFactory2> factory2;
  {
    ComPtr<IDXGIFactory> baseFactory;
    adapter->GetParent(IID_PPV_ARGS(&baseFactory));
    baseFactory.As(&factory2);
  }
  if (!factory2)
    return false;

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
  scd.Flags = 0; // see comments in original

  HRESULT hr_sc = factory2->CreateSwapChainForComposition(
      d3d.Get(), &scd, nullptr, &swapChain);
  if (FAILED(hr_sc)) {
    lg::hrfail(P_E, "CreateSwapChainForComposition", hr_sc);
    return false;
  }

  ComPtr<IDXGIDevice1> dxgiDev1;
  d3d.As(&dxgiDev1);
  if (dxgiDev1 && max_latency > 0)
    dxgiDev1->SetMaximumFrameLatency((UINT)std::min(std::max(1, max_latency), 3));

  if (FAILED(DCompositionCreateDevice(dxgiDev.Get(), IID_PPV_ARGS(&dcompDevice)))) {
    lg::werr(P_E, "DCompositionCreateDevice");
    return false;
  }
  if (FAILED(dcompDevice->CreateTargetForHwnd(hwnd, TRUE, &dcompTarget))) {
    lg::werr(P_E, "CreateTargetForHwnd");
    return false;
  }
  if (FAILED(dcompDevice->CreateVisual(&dcompRoot))) {
    lg::werr(P_E, "CreateVisual");
    return false;
  }
  if (FAILED(dcompRoot->SetContent(swapChain.Get()))) {
    lg::werr(P_E, "SetContent");
    return false;
  }
  if (FAILED(dcompTarget->SetRoot(dcompRoot.Get()))) {
    lg::werr(P_E, "SetRoot");
    return false;
  }
  dcompDevice->Commit();

  if (FAILED(CreateTargets())) {
    lg::werr(P_E, "CreateTargets");
    return false;
  }

  if (FAILED(d2dCtx->CreateSolidColorBrush(
          D2D1::ColorF(1.f, 1.f, 1.f, 1.f), &brush))) {
    lg::werr(P_E, "CreateSolidColorBrush");
    return false;
  }
  d2dCtx->SetAntialiasMode(D2D1_ANTIALIAS_MODE_PER_PRIMITIVE);
  d2dCtx->SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE);

  if (FAILED(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown **>(dwFactory.GetAddressOf())))) {
    lg::werr(P_E, "DWriteCreateFactory");
    return false;
  }

  std::atomic_store_explicit(&current, std::make_shared<DrawList>(), std::memory_order_release);

  return true;
}

void Game_overlay::Impl::DestroyWindowAndDevices() {
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

  if (hwnd) {
    DestroyWindow(hwnd);
    hwnd = nullptr;
  }
  UnregisterClassW(L"GameOverlayDCompWnd", hinst);
}
