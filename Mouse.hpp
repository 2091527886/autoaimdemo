#pragma once

#include <windows.h>
#include "InputSimulator.hpp"

// 仅迁移，不改变逻辑/接口
class Mouse {
public:
    Mouse();
    ~Mouse();

    Mouse(const Mouse&) = delete;
    Mouse& operator=(const Mouse&) = delete;

    // 简洁接口
    void Move(int dx, int dy);

private:
    // 简洁函数指针类型
    using SendInitFn = Send::Error(__stdcall*)(Send::SendType, Send::InitFlags, void*);
    using SendDestroyFn = void(__stdcall*)();
    using SendMouseMoveFn = bool(__stdcall*)(int, int, Send::MoveMode);

    // 简洁成员名（统一后缀 _）
    HMODULE dll_ = nullptr;
    SendInitFn init_ = nullptr;
    SendDestroyFn destroy_ = nullptr;
    SendMouseMoveFn move_ = nullptr;
};
