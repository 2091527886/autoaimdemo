#pragma once
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
#include <windows.h>
#include "Detection.hpp"
#include "PidController.hpp"
#include "Config.hpp"
#include "MouAsync.hpp"

class CtlAsync {
public:
    CtlAsync(const Config& cfg, MouAsync& mou) : c(cfg), mou_(mou) {}
    ~CtlAsync() { stop(); }

    void set(cv::Point ctr, double mfac) {
        ctr_ = ctr; mfac_ = mfac;
        px.setK(c.pid_kp_x, c.pid_ki_x, c.pid_kd_x);
        px.setLim(c.pid_output_limit, c.pid_integral_limit);
        py.setK(c.pid_kp_y, c.pid_ki_y, c.pid_kd_y);
        py.setLim(c.pid_output_limit, c.pid_integral_limit);
    }

    void start();
    void stop();

    void put(const std::vector<Detection>& ds, bool smooth, bool single, double dt);

private:
    void run();

    const Config& c;
    MouAsync& mou_;
    std::thread th_;
    std::atomic<bool> go_{false};
    std::atomic<int> seq_{0};
    HANDLE ev_ = nullptr;

    // geom
    cv::Point ctr_{0,0};
    double mfac_ = 1.0;

    // pid
    PidController px, py;

    struct Msg { std::vector<Detection> ds; bool smooth=false; bool single=false; double dt=0.016; };
    std::mutex mx_;
    Msg msg_;

    // track-keeping
    int cur_tid_ = -1;
};
