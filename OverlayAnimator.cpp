#include "OverlayAnimator.hpp"
#include "Config.hpp"

#include <iostream>
#include <algorithm>
#include <chrono>

OverlayAnimator::OverlayAnimator() = default;
OverlayAnimator::~OverlayAnimator() { stop(); }

bool OverlayAnimator::start(const Config& cfg) {
    if (running_.load()) return true;
    if (!cfg.enable_overlay_animator) return false;
    running_.store(true);
    worker_ = std::thread([this, &cfg]{ this->thread_proc(cfg); });
    return true;
}
void OverlayAnimator::stop() {
    if (!running_.load()) return;
    running_.store(false);
    if (hwnd_) PostMessage(hwnd_, WM_CLOSE, 0, 0);
    if (worker_.joinable()) worker_.join();
    hwnd_ = nullptr;
}
LRESULT CALLBACK OverlayAnimator::WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}
void OverlayAnimator::thread_proc(const Config& cfg) {
    // 线程启动即绑定（如开启）
    if (cfg.enable_thread_affinity && cfg.overlay_thread_affinity_mask) {
        ApplyThreadAffinityMask(cfg.overlay_thread_affinity_mask,
            cfg.enable_optimization_report && cfg.enable_system_opt_report);
    }
    HINSTANCE hInst = GetModuleHandle(NULL);
    const wchar_t* cls = L"RovoOverlayAnimatorWnd";
    WNDCLASSEXW wc{ sizeof(WNDCLASSEXW) };
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = cls;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassExW(&wc);

    int sz = std::max(1, cfg.overlay_pixel_size);
    int x = cfg.overlay_pos_x;
    int y = cfg.overlay_pos_y;
    HWND hWnd = CreateWindowExW(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
                                cls, L"", WS_POPUP, x, y, sz, sz, nullptr, nullptr, hInst, nullptr);
    if (!hWnd) { running_.store(false); return; }
    // Don't activate
    ShowWindow(hWnd, SW_SHOWNA);
    hwnd_ = hWnd;

    // Prepare a 1x1 or 2x2 DIB section for per-pixel alpha updates
    HDC screenDC = GetDC(nullptr);
    HDC memDC = CreateCompatibleDC(screenDC);
    BITMAPV5HEADER bi{};
    bi.bV5Size = sizeof(bi);
    bi.bV5Width = sz; bi.bV5Height = -sz; // top-down
    bi.bV5Planes = 1; bi.bV5BitCount = 32; bi.bV5Compression = BI_BITFIELDS;
    bi.bV5RedMask = 0x00FF0000; bi.bV5GreenMask = 0x0000FF00; bi.bV5BlueMask = 0x000000FF; bi.bV5AlphaMask = 0xFF000000;
    void* bits = nullptr;
    HBITMAP hbmp = CreateDIBSection(screenDC, (BITMAPINFO*)&bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    HGDIOBJ old = SelectObject(memDC, hbmp);

    BLENDFUNCTION bf{ AC_SRC_OVER, 0, 255, AC_SRC_ALPHA };
    SIZE wndSize{ sz, sz };
    POINT srcPos{ 0, 0 };

    // High-frequency timer using waitable timer at 100ns units
    HANDLE hTimer = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (!hTimer) hTimer = CreateWaitableTimerW(NULL, FALSE, NULL);

    // Animation loop: toggle pixel alpha/color every interval
    uint32_t t = 0;
    while (running_.load()) {
        // write pixels
        uint32_t c1 = 0x01010101u * ((t & 1) ? 0x01 : 0x02); // tiny change in alpha/color
        for (int i = 0; i < sz * sz; ++i) {
            ((uint32_t*)bits)[i] = c1;
        }
        POINT wndPos{ x, y };
        UpdateLayeredWindow(hWnd, screenDC, &wndPos, &wndSize, memDC, &srcPos, 0, &bf, ULW_ALPHA);
        ++t;

        // wait for next tick (waitable timer + optional spin to improve accuracy)
        LARGE_INTEGER due;
        due.QuadPart = - (cfg.overlay_update_interval_100ns); // relative, negative, in 100ns units
        uint64_t spin_us = (uint64_t)std::max<long long>(0, cfg.overlay_spin_budget_us);
        if (hTimer && SetWaitableTimer(hTimer, &due, 0, nullptr, nullptr, FALSE)) {
            if (spin_us > 0) {
                // coarse wait then spin for the last few microseconds
                WaitForSingleObject(hTimer, INFINITE);
                auto spin_end = std::chrono::high_resolution_clock::now() + std::chrono::microseconds(spin_us);
                while (std::chrono::high_resolution_clock::now() < spin_end) {
                    // optional pause hint
                    YieldProcessor();
                }
            } else {
                WaitForSingleObject(hTimer, INFINITE);
            }
        } else {
            // Fallback: minimal sleep
            Sleep(0);
        }
    }
    if (hTimer) CloseHandle(hTimer);
    // Cleanup
    SelectObject(memDC, old);
    DeleteObject(hbmp);
    DeleteDC(memDC);
    ReleaseDC(nullptr, screenDC);
    DestroyWindow(hWnd);
}
