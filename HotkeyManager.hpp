#pragma once
#include "Config.hpp"

class HotkeyManager {
public:
    explicit HotkeyManager(const Config& cfg) : cfg_(cfg) {}

    bool SmoothPressed() const;
    bool SingleShotJustPressed();

private:
    const Config& cfg_;
    bool prev_single_shot_pressed_ = false;
};
