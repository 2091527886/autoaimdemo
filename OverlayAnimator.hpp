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
    static LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    void thread_proc(const Config& cfg);
    std::atomic<bool> running_{ false };
    std::thread worker_;
    HWND hwnd_ = nullptr;
};
