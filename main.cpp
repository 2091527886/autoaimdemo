#include "pch.hpp"
#include <iostream>
#include "Log.hpp"
#include <stdexcept>
#include <thread>
#include <atomic>
#include <opencv2/core.hpp>
#include "Config.hpp"
#include "Timing.hpp"
#include "SystemTuner.hpp"
#include "Reporting.hpp"
#include "Detection.hpp"

#include "CtrlHandler.hpp"
#include "Aim.hpp"

int main() {
    SetConsoleCtrlHandler(CtrlHandler, TRUE);

    Config boot_cfg;
    boot_cfg.validate();
    disablePowerThrottle();

    SetProcPrioHigh();
    SetThreadPrioHighest();

    HANDLE hMmcss = nullptr; // MMCSS disabled

   if (boot_cfg.enable_thread_mem_prio) {
       SetThreadMemPrioNormal();
       SetProcPrioBoostDisabled();
   }

    metrics::SystemBootReporting::PrintOverlayBanner(boot_cfg, boot_cfg.ovl);

    metrics::SystemBootReporting::PrintSchedulingBanner(boot_cfg, hMmcss);

    try {
        Aim assistant;
        assistant.loop();
    }
    catch (const std::exception& e) {
        lg::ml(P_F, std::string("Fatal: ") + e.what(), "Check issues.", "Press Enter to exit.");
        std::cin.get();
        g_exit_done.store(true, std::memory_order_relaxed);
        return -1;
    }
    lg::o() << lg::nl << P_I "Done" << lg::endl;
    g_exit_done.store(true, std::memory_order_relaxed);
    return 0;
}
