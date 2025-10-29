#pragma once
#include <cstddef>
#include <string>
#include <tuple>
#include <iostream>
#include "../Config.hpp"
#include <trtyolo.hpp>

namespace metrics {

class TrtPerfReporting {
public:
    explicit TrtPerfReporting(const Config& cfg) : cfg_(cfg) {}

    bool ShouldReport() {
        if (cfg_.enable_performance_report) {
            if (cfg_.perf_report_every_frame) return true;
            perf_frame_counter_++;
            if (cfg_.perf_report_interval_frames <= 1) return true;
            if (perf_frame_counter_ % static_cast<size_t>(cfg_.perf_report_interval_frames) == 0) return true;
        }
        return false;
    }

    void PrintReport(trtyolo::DetectModel& model) {
        auto report = model.performanceReport();
        const std::string& throughput = std::get<0>(report);
        if (!throughput.empty()) {
            const std::string& cpu_latency = std::get<1>(report);
            const std::string& gpu_latency = std::get<2>(report);
            std::cout << "[TRT-YOLO PERF] " << throughput << '\n'
                      << "[TRT-YOLO PERF] " << cpu_latency << '\n'
                      << "[TRT-YOLO PERF] " << gpu_latency << std::endl;
        }
    }

private:
    const Config& cfg_;
    size_t perf_frame_counter_ = 0;
};

} // namespace metrics
