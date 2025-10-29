#ifndef TIMING_HPP
#define TIMING_HPP

#include <vector>
#include <tuple>
#include <limits>
#include <iomanip>
#include <iostream>
#include <algorithm>

// Timing details for detection pipeline
struct TimingDetails {
    double inference_ms = 0;
    double postprocess_ms = 0;
    double total_loop_ms = 0;
};

// Per-frame latency breakdown
struct FrameLatency {
    double capture_ms = 0.0;
    double detect_ms  = 0.0;
    double mouse_ms   = 0.0;
    double loop_ms    = 0.0;
};

// Aggregate and print latency stats over a fixed window
class LatencyAggregator {
public:
    explicit LatencyAggregator(size_t window) : window_(window ? window : 30) {
        buffer_.reserve(window_);
    }

    void add(const FrameLatency& f) {
        buffer_.push_back(f);
        if (buffer_.size() >= window_) {
            print_and_clear();
        }
    }

    void force_print() {
        if (!buffer_.empty()) print_and_clear();
    }

private:
    void print_and_clear() {
        auto [cap_avg, cap_min, cap_max] = stats([](const FrameLatency& f){ return f.capture_ms; });
        auto [det_avg, det_min, det_max] = stats([](const FrameLatency& f){ return f.detect_ms;  });
        auto [mou_avg, mou_min, mou_max] = stats([](const FrameLatency& f){ return f.mouse_ms;   });
        auto [loo_avg, loo_min, loo_max] = stats([](const FrameLatency& f){ return f.loop_ms;    });

        std::cout << std::fixed << std::setprecision(2)
                  << "[LAT] over " << buffer_.size() << " frames"
                  << " | capture ms avg/min/max: " << cap_avg << "/" << cap_min << "/" << cap_max
                  << " | detect: " << det_avg << "/" << det_min << "/" << det_max
                  << " | mouse: "  << mou_avg << "/" << mou_min << "/" << mou_max
                  << " | loop: "   << loo_avg << "/" << loo_min << "/" << loo_max
                  << std::endl;

        buffer_.clear();
    }

    template<typename Getter>
    std::tuple<double,double,double> stats(Getter g) const {
        if (buffer_.empty()) return {0.0, 0.0, 0.0};
        double sum = 0.0, mn = std::numeric_limits<double>::infinity(), mx = 0.0;
        for (const auto& f : buffer_) {
            const double v = g(f);
            sum += v;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        return { sum / static_cast<double>(buffer_.size()), (std::isinf(mn)?0.0:mn), mx };
    }

    size_t window_;
    std::vector<FrameLatency> buffer_;
};

#endif // TIMING_HPP
