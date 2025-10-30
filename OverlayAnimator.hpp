#pragma once

#include <windows.h>
#include <thread>
#include <atomic>

class Config; // forward declaration

class OverlayAnim {
public:
    OverlayAnim();
    ~OverlayAnim();
    bool start(const Config& cfg);
    void stop();
private:
    static LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    static constexpr wchar_t kCls[] = L"RovoOverlayAnimatorWnd";
    void run(const Config& cfg);
    std::atomic<bool> alive{ false };
    std::thread th;
    HWND wnd = nullptr;
};
