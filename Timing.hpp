#ifndef TIMING_HPP
#define TIMING_HPP

#include <vector>
#include <tuple>
#include <limits>
#include "Log.hpp"
#include <algorithm>
#include <string>

// Timing details for detection pipeline
struct Timing {
    double infer_ms = 0;
    double post_ms = 0;
    double loop_ms = 0;
    // TRT perf strings (filled in det thread, printed in main)
    std::string trt_throughput;
    std::string trt_cpu_lat;
    std::string trt_gpu_lat;
};

// Per-frame latency breakdown
struct FrameLat {
    double cap_ms = 0.0;
    double det_ms  = 0.0;
    double mou_ms   = 0.0;
    double trk_ms   = 0.0;
    double ctl_ms   = 0.0;
    double ovl_ms   = 0.0;
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
            printClear();
        }
    }

private:
    void printClear() {
        auto [cap_avg, cap_min, cap_max] = stat([](const FrameLat& f){ return f.cap_ms; });
        auto [det_avg, det_min, det_max] = stat([](const FrameLat& f){ return f.det_ms;  });
        auto [mou_avg, mou_min, mou_max] = stat([](const FrameLat& f){ return f.mou_ms;   });
        auto [trk_avg, trk_min, trk_max] = stat([](const FrameLat& f){ return f.trk_ms;   });
        auto [ctl_avg, ctl_min, ctl_max] = stat([](const FrameLat& f){ return f.ctl_ms;   });
        auto [ovl_avg, ovl_min, ovl_max] = stat([](const FrameLat& f){ return f.ovl_ms;   });
        auto [loo_avg, loo_min, loo_max] = stat([](const FrameLat& f){ return f.loop_ms;    });

        lg::o() << lg::fx2
                  << lg::ovr(P_L, buf.size())
                  << " | cap " << cap_avg << "/" << cap_min << "/" << cap_max
                  << " | det " << det_avg << "/" << det_min << "/" << det_max
                  << " | mou " << mou_avg << "/" << mou_min << "/" << mou_max
                  << " | trk " << trk_avg << "/" << trk_min << "/" << trk_max
                  << " | ctl " << ctl_avg << "/" << ctl_min << "/" << ctl_max
                  << " | ovl " << ovl_avg << "/" << ovl_min << "/" << ovl_max
                  << " | loop " << loo_avg << "/" << loo_min << "/" << loo_max
                  << lg::endl << lg::df;

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
