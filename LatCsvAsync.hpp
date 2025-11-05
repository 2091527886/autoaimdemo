#pragma once

#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <string>
#include <cstdio>
#include <windows.h>
#include "Timing.hpp"
#include "SystemTuner.hpp"
#include "Config.hpp"

// Minimal async CSV writer for per-frame latency; event-driven, non-blocking
class LatCsvAsync {
public:
    LatCsvAsync() = default;
    explicit LatCsvAsync(const Config& cfg) : cfg_(&cfg) {}
    ~LatCsvAsync() { stop(); }

    LatCsvAsync(const LatCsvAsync&) = delete;
    LatCsvAsync& operator=(const LatCsvAsync&) = delete;

    bool start(const std::string& path) {
        bool ex = false;
        if (!go_.compare_exchange_strong(ex, true)) return false;
#if defined(_MSC_VER)
        if (fopen_s(&fp_, path.c_str(), "w") != 0) { fp_ = nullptr; go_.store(false, std::memory_order_relaxed); return false; }
#else
        fp_ = fopen(path.c_str(), "w");
        if (!fp_) { go_.store(false, std::memory_order_relaxed); return false; }
#endif
        // header
        fprintf(fp_, "idx,cap_ms,det_ms,mou_ms,trk_ms,ctl_ms,ovl_ms,loop_ms\n");
        if (!ev_) ev_ = CreateEventW(nullptr, FALSE, FALSE, nullptr); // auto-reset
        th_ = std::thread(&LatCsvAsync::run, this);
        return true;
    }

    void stop() {
        bool ex = true;
        if (go_.compare_exchange_strong(ex, false)) {
            if (ev_) SetEvent(ev_);
            if (th_.joinable()) th_.join();
            if (ev_) { CloseHandle(ev_); ev_ = nullptr; }
            // after worker thread drained, write the average line only (run() already flushed rows)
            if (fp_) {
                if (cnt_ > 0) {
                    double n = static_cast<double>(cnt_);
                    fprintf(fp_, "-1,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                        sum_cap_/n, sum_det_/n, sum_mou_/n, sum_trk_/n, sum_ctl_/n, sum_ovl_/n, sum_loop_/n);
                }
                fflush(fp_);
                fclose(fp_);
                fp_ = nullptr;
            }
        }
    }

    void put(const FrameLat& f) {
        if (!go_.load(std::memory_order_relaxed)) return;
        {
            std::lock_guard<std::mutex> lk(mx_);
            q_.push_back(f);
            // accumulate sums for average line
            sum_cap_ += f.cap_ms;
            sum_det_ += f.det_ms;
            sum_mou_ += f.mou_ms;
            sum_trk_ += f.trk_ms;
            sum_ctl_ += f.ctl_ms;
            sum_ovl_ += f.ovl_ms;
            sum_loop_ += f.loop_ms;
            ++cnt_;
        }
        if (ev_) SetEvent(ev_);
    }

private:
    void run() {
        // optional: set affinity for CSV writer
        if (cfg_ && cfg_->enable_affinity && cfg_->aff_lat) {
            setAffinity(cfg_->aff_lat, cfg_->opt_report && cfg_->opt_report_system);
        }
        while (go_.load(std::memory_order_relaxed)) {
            if (ev_) WaitForSingleObject(ev_, INFINITE);
            if (!go_.load(std::memory_order_relaxed)) break;
            std::vector<FrameLat> cur;
            {
                std::lock_guard<std::mutex> lk(mx_);
                cur.swap(q_);
            }
            if (fp_) {
                for (const auto& f : cur) {
                    // keep csv short & readable
                    fprintf(fp_, "%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                        (unsigned long long)f.idx,
                        f.cap_ms, f.det_ms, f.mou_ms, f.trk_ms, f.ctl_ms, f.ovl_ms, f.loop_ms);
                }
                fflush(fp_);
            }
        }
        // drain leftover if any (keep q_ for stop() to also flush and then print avg)
        if (fp_) {
            for (const auto& f : q_) {
                fprintf(fp_, "%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                    (unsigned long long)f.idx,
                    f.cap_ms, f.det_ms, f.mou_ms, f.trk_ms, f.ctl_ms, f.ovl_ms, f.loop_ms);
            }
            fflush(fp_);
        }
    }

    std::atomic<bool> go_{false};
    std::thread th_;
    HANDLE ev_ = nullptr;
    std::mutex mx_;
    std::vector<FrameLat> q_;
    FILE* fp_ = nullptr;

    const Config* cfg_ = nullptr;

    // aggregates for averages
    double sum_cap_ = 0.0, sum_det_ = 0.0, sum_mou_ = 0.0, sum_trk_ = 0.0, sum_ctl_ = 0.0, sum_ovl_ = 0.0, sum_loop_ = 0.0;
    unsigned long long cnt_ = 0;
};
