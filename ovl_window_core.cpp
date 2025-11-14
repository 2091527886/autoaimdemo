#include "ovl_impl.hpp"
#include "ovl_util.hpp"
#include "Config.hpp"
#include "pch.hpp"

bool Game_overlay::Impl::Start() {
  if (running.load()) return true;
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
  if (!running.exchange(false)) return;
  if (hwnd) PostMessageW(hwnd, WM_CLOSE, 0, 0);
  DestroyWindowAndDevices();
  CoUninitialize();
}

LRESULT CALLBACK Game_overlay::Impl::WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  Impl* self = reinterpret_cast<Impl*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));
  switch (msg) {
  case WM_NCCREATE: {
    auto cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
    SetWindowLongPtrW(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
    return TRUE;
  }
  case WM_NCHITTEST: return HTTRANSPARENT;
  case WM_MOUSEACTIVATE: return MA_NOACTIVATE;
  case WM_SIZE:
    if (self && self->swapChain) {
      UINT w = LOWORD(lParam), h = HIWORD(lParam);
      if (w > 0 && h > 0) {
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
