#include "Aim.hpp"
#include "Log.hpp"
#include "sh.hpp"

void Aim::log_detections(bool frame_available) {
    detection_frame_since_log_ = detection_frame_since_log_ || frame_available;
    const auto now = std::chrono::steady_clock::now();
    if (now - last_detection_log_ < std::chrono::seconds(1)) return;
    last_detection_log_ = now;

    if (!detection_frame_since_log_) {
        LO << "[DETECT] frame unavailable" << END;
    } else if (!c.pose_enable) {
        LO << "[DETECT] pose disabled" << END;
    } else if (!pose.ready()) {
        LO << "[DETECT] pose model unavailable" << END;
    } else if (!pose.has_result()) {
        LO << "[DETECT] inference result unavailable" << END;
    } else {
        size_t enemy = 0, teammate = 0;
        for (int class_id : pose.result_classes()) {
            if (class_id == 0) ++enemy;
            else if (class_id == 1) ++teammate;
        }
        LO << "[DETECT] enemy=" << enemy << " | teammate=" << teammate << END;
    }
    detection_frame_since_log_ = false;
}

void Aim::log_lat(FrameLat& fl) {
    if (!(c.timing && c.log_latency)) return;
    lat.push(fl);
    ++fcnt;
    int n = MX(1, c.log_latency_every_n);
    if (fcnt % (size_t)n == 0){
        size_t drop_now = fc.drop - drop_prev;
        LO << lg::fx2 << P_L "[fps] " << "drop=" << drop_now << " | ovr=" << fc.ovr << "ms";
        if (mvq_events_>0) {
            double avg_loss = mvq_loss_sum_ / (double)mvq_events_;
            LO << " | mvq_loss_sum=" << mvq_loss_sum_ << " | mvq_ev=" << mvq_events_ << " | mvq_avg=" << avg_loss;
            mvq_loss_sum_ = 0.0;
            mvq_events_ = 0;
        }
        LO << END << lg::df;
        drop_prev = fc.drop;
    }
}

void Aim::trt_log() {
    if (c.timing && !tm.trt_throughput.empty()) {
        int n = MX(1, c.log_latency_every_n);
        if ((c.frame_idx % n) == 0) {
            lg::ml(P_T, tm.trt_throughput, tm.trt_cpu_lat, tm.trt_gpu_lat);
            tm.trt_throughput.clear();
            tm.trt_cpu_lat.clear();
            tm.trt_gpu_lat.clear();
        }
    }
}
