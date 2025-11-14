#pragma once

#include <windows.h>
#include "InputSimulator.hpp"

class Mouse {
public:
    Mouse();
    ~Mouse();

    Mouse(const Mouse&) = delete;
    Mouse& operator=(const Mouse&) = delete;

        void Move(int dx, int dy);

private:
        using SendInitFn = Send::Error(__stdcall*)(Send::SendType, Send::InitFlags, void*);
    using SendDestroyFn = void(__stdcall*)();
    using SendMouseMoveFn = bool(__stdcall*)(int, int, Send::MoveMode);

        HMODULE dll_ = nullptr;
    SendInitFn init_ = nullptr;
    SendDestroyFn destroy_ = nullptr;
    SendMouseMoveFn move_ = nullptr;
};
