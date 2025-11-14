#include "HotkeyManager.hpp"
#include <windows.h>

bool HotkeyManager::SmoothPressed() const {
    return (GetAsyncKeyState(cfg_.aim_key) & 0x8000) != 0;
}

bool HotkeyManager::SingleShotJustPressed() {
    bool is_pressed = (GetAsyncKeyState(cfg_.test_move_key) & 0x8000) != 0;
    bool just_pressed = is_pressed && !prev_single_shot_pressed_;
    prev_single_shot_pressed_ = is_pressed;
    return just_pressed;
}
