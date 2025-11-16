#pragma once
#include "Config.hpp"

class HotkeyManager {
public:
    explicit HotkeyManager(const Config& cfg) : cfg_(cfg) {}

    bool SmoothPressed() const;
private:
    const Config& cfg_;
};
