#pragma once // 防止头文件被重复包含

#include <windows.h>
#include <cstdint> // 包含这个头文件以使用 uint32_t

// 声明我们将要使用的枚举
// 注意：保持命名空间与枚举不变，避免影响外部调用
namespace Send {
    enum class Error {
        Success = 0, InvalidArgument = 1, DeviceNotFound = 2, DriverError = 3
    };
    enum class SendType {
        SendInput = 0, Logitech = 1, Razer = 2, DD = 3,
        MouClassInputInjection = 4, LogitechGHubNew = 5, AnyDriver = 100
    };
    enum class MoveMode {
        Absolute = 0, Relative = 1
    };
    enum class InitFlags {
        // ... 如果有的话
    };
}

// 仅改名，不改变逻辑/实现；该类当前未在外部使用
class ISim {
public:
    ISim();
    ~ISim();

    Send::Error init(Send::SendType type);
    // 相对移动
    void move(int dx, int dy);
    // 绝对移动接口暂不提供（原注释保留意图）
    // void moveAbs(int x, int y);

private:
    HMODULE dll_;

    // 函数指针类型（简洁命名）
    using SendInitFn = Send::Error(__stdcall*)(Send::SendType, Send::InitFlags, void*);
    using SendDestroyFn = void(__stdcall*)();
    using SendMouseMoveFn = bool(__stdcall*)(int, int, Send::MoveMode);

    // 函数指针成员
    SendInitFn init_;
    SendDestroyFn destroy_;
    SendMouseMoveFn move_;

    bool ready_;
};
