#ifndef TIMING_HPP
#define TIMING_HPP

#include <vector>
#include <tuple>
#include <limits>
#include <iomanip>
#include <iostream>
#include <algorithm>

// Timing details for detection pipeline
struct Timing {
    double infer_ms = 0;
    double post_ms = 0;
    double loop_ms = 0;
};

// Per-frame latency breakdown
struct FrameLat {
    double cap_ms = 0.0;
    double det_ms  = 0.0;
    double mou_ms   = 0.0;
    double loop_ms    = 0.0;
};

// Aggregate and print latency stats over a fixed window
class LatAgg {
public:
    explicit LatAgg(size_t window) : wnd(window ? window : 30) {
        buf.reserve(wnd);
    }

    void push(const FrameLat& f) {
        buf.push_back(f);
        if (buf.size() >= wnd) {
            print_and_clear();
        }
    }

    void flush() {
        if (!buf.empty()) printClear();
    }

private:
    void printClear() {
        auto [cap_avg, cap_min, cap_max] = stat([](const FrameLat& f){ return f.cap_ms; });
        auto [det_avg, det_min, det_max] = stat([](const FrameLat& f){ return f.det_ms;  });
        auto [mou_avg, mou_min, mou_max] = stat([](const FrameLat& f){ return f.mou_ms;   });
        auto [loo_avg, loo_min, loo_max] = stat([](const FrameLat& f){ return f.loop_ms;    });

        std::cout << std::fixed << std::setprecision(2)
                  << "[LAT] over " << buf.size() << " frames"
                  << " | capture ms avg/min/max: " << cap_avg << "/" << cap_min << "/" << cap_max
                  << " | detect: " << det_avg << "/" << det_min << "/" << det_max
                  << " | mouse: "  << mou_avg << "/" << mou_min << "/" << mou_max
                  << " | loop: "   << loo_avg << "/" << loo_min << "/" << loo_max
                  << std::endl;

        buf.clear();
    }

    template<typename Getter>
    std::tuple<double,double,double> stat(Getter g) const {
        if (buf.empty()) return {0.0, 0.0, 0.0};
        double sum = 0.0, mn = std::numeric_limits<double>::infinity(), mx = 0.0;
        for (const auto& f : buf) {
            const double v = g(f);
            sum += v;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        return { sum / static_cast<double>(buf.size()), (std::isinf(mn)?0.0:mn), mx };
    }

    size_t wnd;
    std::vector<FrameLat> buf;
};

#endif // TIMING_HPP
