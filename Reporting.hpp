#pragma once
#include <cstddef>
#include <iostream>
#include "Log.hpp"
#include <iomanip>
#include <algorithm>
#include <string>
#include <tuple>

#include "Config.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

#include <trtyolo.hpp>

namespace metrics {

class CaptureReporting {
public:
    explicit CaptureReporting(const Config& cfg) : cfg_(cfg) {}

    void BootBanner(size_t stagingCount) const {
        if (cfg_.opt_report && cfg_.opt_report_capture) {
            lg::o() << "[BOOT] Capture pipeline: "
                      << "dirty-skip=ON"
                      << " | staging-buffers=" << stagingCount
                      << " | event-query=ON"
                      << " | map=DO_NOT_WAIT"
                      << " | cache-reuse=ON"
                      << std::endl;
            lg::o() << "[BOOT] Timing: "
                      << "step-timing=" << (cfg_.capture_timing ? "ON" : "OFF")
                      << " | interval=" << cfg_.capture_timing_every_n
                      << " | success-only=" << (cfg_.capture_timing_on_success ? "ON" : "OFF")
                      << std::endl;
        }
    }

    void ReportStepTimings(double t_acquire_ms, double t_qi_ms, double t_copy_ms,
                           double t_map_ms, double t_cvt_ms, double t_unmap_ms, double t_total_ms,
                           size_t& stat_dirty_skips_member,
                           size_t& stat_event_not_ready_member,
                           size_t& stat_map_still_drawing_member,
                           size_t& stat_cache_reused_member,
                           size_t& stat_success_member)
    {
        // Merge member counters into period counters, then clear member ones
        stat_dirty_skips_       += stat_dirty_skips_member;
        stat_event_not_ready_   += stat_event_not_ready_member;
        stat_map_still_drawing_ += stat_map_still_drawing_member;
        stat_cache_reused_      += stat_cache_reused_member;
        stat_success_           += stat_success_member;
        stat_dirty_skips_member = stat_event_not_ready_member = stat_map_still_drawing_member = stat_cache_reused_member = stat_success_member = 0;

        opt_counter_++;
        counter_++;
        acc_acq_   += t_acquire_ms;
        acc_qi_    += t_qi_ms;
        acc_copy_  += t_copy_ms;
        acc_map_   += t_map_ms;
        acc_cvt_   += t_cvt_ms;
        acc_unmap_ += t_unmap_ms;
        acc_total_ += t_total_ms;

        bool should_print = cfg_.capture_timing_each_frame;
        if (!should_print) {
            int interval = std::max(1, cfg_.capture_timing_every_n);
            if (interval <= 1) should_print = true;
            else if (counter_ % static_cast<size_t>(interval) == 0) should_print = true;
        }
        if (should_print) {
            double n = static_cast<double>(counter_);
            lg::o() << std::fixed << std::setprecision(3)
                      << "[CAPTURE PERF] over " << counter_ << " frames"
                      << " | acquire=" << (acc_acq_ / n) << " ms"
                      << " | qi="      << (acc_qi_  / n) << " ms"
                      << " | copy="    << (acc_copy_/ n) << " ms"
                      << " | map="     << (acc_map_ / n) << " ms"
                      << " | cvt="     << (acc_cvt_ / n) << " ms"
                      << " | unmap="   << (acc_unmap_/n) << " ms"
                      << " | total="   << (acc_total_/n) << " ms"
                      << std::endl;

            if (cfg_.opt_report && cfg_.opt_report_capture) {
                bool should_print_opt = false;
                int opt_interval = std::max(1, cfg_.opt_report_every_n);
                if (opt_interval <= 1) should_print_opt = true;
                else if (opt_counter_ % static_cast<size_t>(opt_interval) == 0) should_print_opt = true;
                if (should_print_opt) {
                    size_t total = static_cast<size_t>(opt_counter_);
                    auto pct = [&](size_t v){ return total ? (100.0 * double(v) / double(total)) : 0.0; };
                    lg::o() << std::fixed << std::setprecision(1)
                              << "[CAPTURE OPT] over " << total << " frames"
                              << " | dirty-skip="        << stat_dirty_skips_        << " (" << pct(stat_dirty_skips_)        << "%)"
                              << " | event-not-ready="   << stat_event_not_ready_    << " (" << pct(stat_event_not_ready_)    << "%)"
                              << " | map-still-drawing=" << stat_map_still_drawing_  << " (" << pct(stat_map_still_drawing_)  << "%)"
                              << " | cache-reused="      << stat_cache_reused_       << " (" << pct(stat_cache_reused_)       << "%)"
                              << " | success="           << stat_success_            << " (" << pct(stat_success_)            << "%)"
                              << std::endl;
                    opt_counter_ = 0;
                    stat_dirty_skips_ = stat_event_not_ready_ = stat_map_still_drawing_ = stat_cache_reused_ = stat_success_ = 0;
                }
            }

            // reset perf accumulators
            counter_ = 0;
            acc_acq_ = acc_qi_ = acc_copy_ = acc_map_ = acc_cvt_ = acc_unmap_ = acc_total_ = 0.0;
        }
    }

private:
    const Config& cfg_;
    size_t opt_counter_ = 0;
    size_t stat_dirty_skips_ = 0, stat_event_not_ready_ = 0, stat_map_still_drawing_ = 0, stat_cache_reused_ = 0, stat_success_ = 0;

    size_t counter_ = 0;
    double acc_acq_ = 0.0, acc_qi_ = 0.0, acc_copy_ = 0.0, acc_map_ = 0.0, acc_cvt_ = 0.0, acc_unmap_ = 0.0, acc_total_ = 0.0;
};

struct SystemBootReporting {
    static void PrintOverlayBanner(const Config& cfg, bool started) {
        if (!cfg.opt_report || !cfg.opt_report_system) return;
        if (started) {
            lg::o() << "[BOOT] Overlay animator=ON (size=" << cfg.overlay_px
                      << ", interval=" << cfg.overlay_interval_100ns << "x100ns, spin="
                      << cfg.overlay_spin_us << "us, pos="
                      << cfg.overlay_x << "," << cfg.overlay_y << ")" << std::endl;
        } else {
            lg::o() << "[BOOT] Overlay animator=OFF" << std::endl;
        }
    }

#ifdef _WIN32
    static void PrintSchedulingBanner(const Config& cfg, HANDLE hMmcss) {
        if (!cfg.opt_report || !cfg.opt_report_system) return;
        lg::wo() << L"[BOOT] Scheduling: MMCSS=" << (hMmcss?L"ON":L"OFF")
                   << L" (task=" << cfg.mmcss_task << L", prio=" << (int)cfg.mmcss_prio << L")" << std::endl;
    }
#endif
};

class TrtPerfReporting {
public:
    explicit TrtPerfReporting(const Config& cfg) : cfg_(cfg) {}

    bool ShouldReport() {
        if (cfg_.perf_report) {
            if (cfg_.perf_each_frame) return true;
            perf_frame_counter_++;
            if (cfg_.perf_report_every_n <= 1) return true;
            if (perf_frame_counter_ % static_cast<size_t>(cfg_.perf_report_every_n) == 0) return true;
        }
        return false;
    }

    void PrintReport(trtyolo::DetectModel& model) {
        auto report = model.performanceReport();
        const std::string& throughput = std::get<0>(report);
        if (!throughput.empty()) {
            const std::string& cpu_latency = std::get<1>(report);
            const std::string& gpu_latency = std::get<2>(report);
            lg::o() << "[TRT-YOLO PERF] " << throughput << '\n'
                      << "[TRT-YOLO PERF] " << cpu_latency << '\n'
                      << "[TRT-YOLO PERF] " << gpu_latency << std::endl;
        }
    }

private:
    const Config& cfg_;
    size_t perf_frame_counter_ = 0;
};

} 