#include <string>
#include "Mouse.hpp"

// 仅迁移，不改变逻辑/实现细节
Mouse::Mouse() {
    std::cout << "[Mouse] Initializing driver..." << std::endl;
    dll_ = LoadLibrary(L"IbInputSimulator.dll");
    if (!dll_) {
        throw std::runtime_error("[Mouse] Load IbInputSimulator.dll failed");
    }

    init_ = (SendInitFn)GetProcAddress(dll_, "IbSendInit");
    destroy_ = (SendDestroyFn)GetProcAddress(dll_, "IbSendDestroy");
    move_ = (SendMouseMoveFn)GetProcAddress(dll_, "IbSendMouseMove");

    if (!init_ || !destroy_ || !move_) {
        FreeLibrary(dll_);
        dll_ = nullptr;
        throw std::runtime_error("[Mouse] Required functions not found in IbInputSimulator.dll");
    }
    // Logitech SendInput
    Send::Error error = init_(Send::SendType::Logitech, static_cast<Send::InitFlags>(0), nullptr);
    if (error != Send::Error::Success) {
        FreeLibrary(dll_);
        dll_ = nullptr;
        throw std::runtime_error("[Mouse] Init failed: " + std::to_string(static_cast<int>(error)));
    }
    std::cout << "[Mouse] Driver ready" << std::endl;
}

Mouse::~Mouse() {
    if (dll_) {
        if (destroy_) destroy_();
        FreeLibrary(dll_);
        std::cout << "[Mouse] Driver cleaned" << std::endl;
    }
}

void Mouse::Move(int dx, int dy) {
    if (move_) {
        bool ok = move_(dx, dy, Send::MoveMode::Relative);
        if (!ok) {
            std::cerr << "[Mouse] Move failed dx=" << dx << " dy=" << dy << std::endl;
        }
    }
}
