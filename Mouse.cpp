#include <string>
#include "Mouse.hpp"
#include "Log.hpp"

// 仅迁移，不改变逻辑/实现细节
Mouse::Mouse() {
    lg::o() << P_M "Init driver" << lg::endl;
    dll_ = LoadLibrary(L"IbInputSimulator.dll");
    if (!dll_) {
        throw std::runtime_error(P_M "Load IbInputSimulator.dll failed");
    }

    init_ = (SendInitFn)GetProcAddress(dll_, "IbSendInit");
    destroy_ = (SendDestroyFn)GetProcAddress(dll_, "IbSendDestroy");
    move_ = (SendMouseMoveFn)GetProcAddress(dll_, "IbSendMouseMove");

    if (!init_ || !destroy_ || !move_) {
        FreeLibrary(dll_);
        dll_ = nullptr;
        throw std::runtime_error(P_M "Required functions not found in IbInputSimulator.dll");
    }
    // Logitech SendInput
    Send::Error error = init_(Send::SendType::Logitech, static_cast<Send::InitFlags>(0), nullptr);
    if (error != Send::Error::Success) {
        FreeLibrary(dll_);
        dll_ = nullptr;
        throw std::runtime_error(P_M "Init failed: " + std::to_string(static_cast<int>(error)));
    }
    lg::o() << P_M "Driver OK" << lg::endl;
}

Mouse::~Mouse() {
    if (dll_) {
        if (destroy_) destroy_();
        FreeLibrary(dll_);
        lg::o() << P_M "Driver bye" << lg::endl;
    }
}

void Mouse::Move(int dx, int dy) {
    if (move_) {
        bool ok = move_(dx, dy, Send::MoveMode::Relative);
        // 仅在失败时日志，避免影响 mou_ms 计时与日志噪声
        if (!ok) {
            lg::ok_or_werr_msg(false, P_M, std::string("mv dx=") + std::to_string(dx) + " dy=" + std::to_string(dy), ok);
        }
    }
}
