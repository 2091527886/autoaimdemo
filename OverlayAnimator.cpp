#include "OverlayAnimator.hpp"
#include "Config.hpp"
#include "SystemTuner.hpp"

#include <algorithm>
#include <chrono>

OverlayAnim::OverlayAnim() = default;
OverlayAnim::~OverlayAnim() { stop(); }

bool OverlayAnim::start(const Config& cfg) {
    if (alive.load()) return true;
    if (!cfg.overlay_anim) return false;
    alive.store(true);
    th = std::thread([this, &cfg]{ this->run(cfg); });
    return true;
}
void OverlayAnim::stop() {
    if (!alive.load()) return;
    alive.store(false);
    if (wnd) PostMessage(wnd, WM_CLOSE, 0, 0);
    if (th.joinable()) th.join();
    wnd = nullptr;
}
LRESULT CALLBACK OverlayAnim::WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}
void OverlayAnim::run(const Config& cfg) {
    // 线程启动即绑定（如开启）
    if (cfg.enable_affinity && cfg.aff_ovl) {
        setAffinity(cfg.aff_ovl,
            cfg.opt_report && cfg.opt_report_system);
    }
    HINSTANCE hInst = GetModuleHandle(NULL);
    const wchar_t* cls = kCls;
    WNDCLASSEXW wc{ sizeof(WNDCLASSEXW) };
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = cls;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassExW(&wc);

    int size = std::max(1, cfg.overlay_px);
    int x = cfg.overlay_x;
    int y = cfg.overlay_y;
    HWND hWnd = CreateWindowExW(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
                                cls, L"", WS_POPUP, x, y, size, size, nullptr, nullptr, hInst, nullptr);
    if (!hWnd) { alive.store(false); return; }
    // Don't activate
    ShowWindow(hWnd, SW_SHOWNA);
    wnd = hWnd;

    // Prepare a 1x1 or 2x2 DIB section for per-pixel alpha updates
    HDC sdc = GetDC(nullptr);
    HDC mdc = CreateCompatibleDC(sdc);
    BITMAPV5HEADER bi{};
    bi.bV5Size = sizeof(bi);
    bi.bV5Width = size; bi.bV5Height = -size; // top-down
    bi.bV5Planes = 1; bi.bV5BitCount = 32; bi.bV5Compression = BI_BITFIELDS;
    bi.bV5RedMask = 0x00FF0000; bi.bV5GreenMask = 0x0000FF00; bi.bV5BlueMask = 0x000000FF; bi.bV5AlphaMask = 0xFF000000;
    void* pixels = nullptr;
    HBITMAP bmp = CreateDIBSection(sdc, (BITMAPINFO*)&bi, DIB_RGB_COLORS, &pixels, nullptr, 0);
    HGDIOBJ old = SelectObject(mdc, bmp);

    BLENDFUNCTION blend{ AC_SRC_OVER, 0, 255, AC_SRC_ALPHA };
    SIZE winSize{ size, size };
    POINT src{ 0, 0 };

    // High-frequency timer using waitable timer at 100ns units
    HANDLE timer = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (!timer) timer = CreateWaitableTimerW(NULL, FALSE, NULL);

    // Animation loop: toggle pixel alpha/color every interval
    uint32_t tick = 0;
    while (alive.load()) {
        // write pixels
        uint32_t c1 = 0x01010101u * ((tick & 1) ? 0x01 : 0x02); // tiny change in alpha/color
        for (int i = 0; i < size * size; ++i) {
            ((uint32_t*)pixels)[i] = c1;
        }
        POINT winPos{ x, y };
        UpdateLayeredWindow(hWnd, sdc, &winPos, &winSize, mdc, &src, 0, &blend, ULW_ALPHA);
        ++tick;

        // wait for next tick (waitable timer + optional spin to improve accuracy)
        LARGE_INTEGER due;
        due.QuadPart = - (cfg.overlay_interval_100ns); // relative, negative, in 100ns units
        uint64_t spin_us = (uint64_t)std::max<long long>(0, cfg.overlay_spin_us);
        if (timer && SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
            if (spin_us > 0) {
                // coarse wait then spin for the last few microseconds
                WaitForSingleObject(timer, INFINITE);
                auto spin_end = std::chrono::high_resolution_clock::now() + std::chrono::microseconds(spin_us);
                while (std::chrono::high_resolution_clock::now() < spin_end) {
                    // optional pause hint
                    YieldProcessor();
                }
            } else {
                WaitForSingleObject(timer, INFINITE);
            }
        } else {
            // Fallback: minimal sleep
            Sleep(0);
        }
    }
    if (timer) CloseHandle(timer);
    // Cleanup
    SelectObject(mdc, old);
    DeleteObject(bmp);
    DeleteDC(mdc);
    ReleaseDC(nullptr, sdc);
    DestroyWindow(hWnd);
}
