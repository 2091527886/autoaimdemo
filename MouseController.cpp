#include <string>
#include "MouseController.hpp"

// 仅迁移，不改变逻辑/实现细节
MouseController::MouseController() {
    std::cout << "--- Initializing Mouse Simulation Driver... ---" << std::endl;
    hMouseDll = LoadLibrary(L"IbInputSimulator.dll");
    if (!hMouseDll) {
        throw std::runtime_error("Failed to load IbInputSimulator.dll! Ensure it's in the EXE directory.");
    }

    IbSendInit_ptr = (pIbSendInit)GetProcAddress(hMouseDll, "IbSendInit");
    IbSendDestroy_ptr = (pIbSendDestroy)GetProcAddress(hMouseDll, "IbSendDestroy");
    IbSendMouseMove_ptr = (pIbSendMouseMove)GetProcAddress(hMouseDll, "IbSendMouseMove");

    if (!IbSendInit_ptr || !IbSendDestroy_ptr || !IbSendMouseMove_ptr) {
        FreeLibrary(hMouseDll);
        hMouseDll = nullptr;
        throw std::runtime_error("Failed to find required functions in IbInputSimulator.dll.");
    }
    // Logitech SendInput
    Send::Error error = IbSendInit_ptr(Send::SendType::Logitech, static_cast<Send::InitFlags>(0), nullptr);
    if (error != Send::Error::Success) {
        FreeLibrary(hMouseDll);
        hMouseDll = nullptr;
        throw std::runtime_error("Failed to initialize mouse driver! Code: " + std::to_string(static_cast<int>(error)) + ". Run as Administrator.");
    }
    std::cout << "--- Mouse driver initialized successfully! ---" << std::endl;
}

MouseController::~MouseController() {
    if (hMouseDll) {
        if (IbSendDestroy_ptr) IbSendDestroy_ptr();
        FreeLibrary(hMouseDll);
        std::cout << "--- Mouse driver resources cleaned up. ---" << std::endl;
    }
}

void MouseController::MoveRelative(int dx, int dy) {
    if (IbSendMouseMove_ptr) {
        bool ok = IbSendMouseMove_ptr(dx, dy, Send::MoveMode::Relative);
        if (!ok) {
            std::cerr << "[ERROR] IbSendMouseMove failed (dx=" << dx << ", dy=" << dy << ")" << std::endl;
        }
    }
}
