#pragma once

#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <string>
#include <cstdio>
#include <windows.h>
#include "Timing.hpp"

// Minimal async CSV writer for per-frame latency; event-driven, non-blocking
class LatCsvAsync {
public:
    LatCsvAsync() = default;
    ~LatCsvAsync() { stop(); }

    LatCsvAsync(const LatCsvAsync&) = delete;
    LatCsvAsync& operator=(const LatCsvAsync&) = delete;

    bool start(const std::string& path) {
        bool ex = false;
        if (!go_.compare_exchange_strong(ex, true)) return false;
        fp_ = fopen(path.c_str(), "w");
        if (!fp_) { go_.store(false, std::memory_order_relaxed); return false; }
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
            if (fp_) { fflush(fp_); fclose(fp_); fp_ = nullptr; }
        }
    }

    void put(const FrameLat& f) {
        if (!go_.load(std::memory_order_relaxed)) return;
        {
            std::lock_guard<std::mutex> lk(mx_);
            q_.push_back(f);
        }
        if (ev_) SetEvent(ev_);
    }

private:
    void run() {
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
        // drain leftover if any
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
};
