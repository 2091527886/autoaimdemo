#pragma once
#include <windows.h>
#include <iostream>
#include <string>
#include "../Config.hpp"

namespace metrics {

struct SystemBootReporting {
    static void PrintOverlayBanner(const Config& cfg, bool started) {
        if (!cfg.enable_optimization_report || !cfg.enable_system_opt_report) return;
        if (started) {
            std::cout << "[BOOT] Overlay animator=ON (size=" << cfg.overlay_pixel_size
                      << ", interval=" << cfg.overlay_update_interval_100ns << "x100ns, spin="
                      << cfg.overlay_spin_budget_us << "us, pos="
                      << cfg.overlay_pos_x << "," << cfg.overlay_pos_y << ")" << std::endl;
        } else {
            std::cout << "[BOOT] Overlay animator=OFF" << std::endl;
        }
    }

    static void PrintSchedulingBanner(const Config& cfg, HANDLE hMmcss) {
        if (!cfg.enable_optimization_report || !cfg.enable_system_opt_report) return;
        std::wcout << L"[BOOT] Scheduling: MMCSS=" << (hMmcss?L"ON":L"OFF")
                   << L" (task=" << cfg.mmcss_task_name << L", prio=" << (int)cfg.mmcss_priority << L")" << std::endl;
    }
};

} // namespace metrics
