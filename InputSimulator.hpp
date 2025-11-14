#pragma once // 防止头文件被重复包含

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

