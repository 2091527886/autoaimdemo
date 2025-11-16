#include "HotkeyManager.hpp"
bool HotkeyManager::SmoothPressed() const {
    return (GetAsyncKeyState(cfg_.aim_key) & 0x8000) != 0;
}
