#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <random>
#include <chrono>
#include "Config.hpp"
#include "Timing.hpp"
#include "Detection.hpp"
#include "Mouse.hpp"
#include "AimAlgo.hpp"
#include "HotkeyManager.hpp"
#include "ScreenCapturer.hpp"
#include "ObjDet.hpp"
#include "Game_overlay.h"
#include "FpsCtl.hpp"

class Aim {
private:
    struct WindState {
        cv::Point2d residual{0.0, 0.0};
        cv::Point2d v{0.0, 0.0};
        std::mt19937 rng{std::random_device{}()};
        std::normal_distribution<double> n01{0.0, 1.0};
    };

public:
    // FpsCtl 独立为全局结构，保持名称不变
    

    Aim();
    ~Aim();
    void loop();

private:
    // helpers
    double get_dt();
    double anch(double h);
    void set_seg(double h);
    std::pair<double,double> pred_err(const Detection& d, double dt, bool enable_pred);
    bool capture(FrameLat& fl, ScreenCapturer::CapturedView& view);
    bool infer_view(ScreenCapturer::CapturedView& view);
    void mv(int mx, int my, double& ms);
    void mv_wind(int mx, int my, double& ms);
    void draw_ovl(double& ms);
    void log_lat(FrameLat& fl);
    void trt_log();

    // state
    Config c;
    ScreenCapturer cap;
    ObjDet od;
    Mouse mouse;
    AimAlgo algo;
    HotkeyManager hk;
    const cv::Rect crop;
    const cv::Point ctr;
    double mfac = 1.0;
    std::vector<Detection> ds;
    Timing tm;
    LatAgg lat;

    FpsCtl fc;
    size_t fcnt = 0;
    size_t drop_prev = 0;

    std::chrono::high_resolution_clock::time_point last_dt_ts = std::chrono::high_resolution_clock::now();

    Game_overlay gover;

    cv::Point2d v_lp_ {0.0, 0.0};
    cv::Point2d last_tp_ {0.0, 0.0};
    bool last_tp_valid_ = false;
    int  last_seg_ = 1; // 0:far,1:mid,2:near

    WindState wind_;
};
