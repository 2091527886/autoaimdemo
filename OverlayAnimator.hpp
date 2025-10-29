#pragma once

#include <windows.h>
#include <thread>
#include <atomic>

class Config; // forward declaration

class OverlayAnimator {
public:
    OverlayAnimator();
    ~OverlayAnimator();
    bool start(const Config& cfg);
    void stop();
private:
    static LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    static constexpr wchar_t kClass[] = L"RovoOverlayAnimatorWnd";
    void loop(const Config& cfg);
    std::atomic<bool> running{ false };
    std::thread thread;
    HWND hwnd = nullptr;
};
