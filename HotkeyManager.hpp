#pragma once
#include <windows.h>
#include "Config.hpp"

class HotkeyManager {
public:
    explicit HotkeyManager(const Config& cfg) : cfg_(cfg) {}

        bool SmoothPressed() const {
        return (GetAsyncKeyState(cfg_.aim_key) & 0x8000) != 0;
    }

    // 边沿触发
    bool SingleShotJustPressed() {
        bool is_pressed = (GetAsyncKeyState(cfg_.test_move_key) & 0x8000) != 0;
        bool just_pressed = is_pressed && !prev_single_shot_pressed_;
        prev_single_shot_pressed_ = is_pressed;
        return just_pressed;
    }

private:
    const Config& cfg_;
    bool prev_single_shot_pressed_ = false; // 等价于 main.cpp 中的 static f8_was_pressed
};
