#include "Aim.hpp"
#include "Log.hpp"

void Aim::log_lat(FrameLat& fl) {
    if (!(c.timing && c.log_latency)) return;
    lat.push(fl);
    ++fcnt;
    int n = std::max(1, c.log_latency_every_n);
    if (fcnt % (size_t)n == 0){ size_t drop_now = fc.drop - drop_prev; LO << lg::fx2 << P_L "[fps] " << "drop=" << drop_now << " | ovr=" << fc.ovr << "ms" << END << lg::df; drop_prev = fc.drop; }
}

void Aim::trt_log() {
    if (c.timing && !tm.trt_throughput.empty()) {
        lg::ml(P_T, tm.trt_throughput, tm.trt_cpu_lat, tm.trt_gpu_lat);
        tm.trt_throughput.clear();
        tm.trt_cpu_lat.clear();
        tm.trt_gpu_lat.clear();
    }
}
