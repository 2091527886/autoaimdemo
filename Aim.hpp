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
#include "Pose.hpp"
#include "OCSort.hpp"
#include "target_selector.hpp"
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
    double anch_read(double h) const; // read-only anchor, no state write
    void set_seg(double h);
    std::pair<double,double> pred_err(const Detection& d, double dt, bool enable_pred);
    bool capture(FrameLat& fl, ScreenCapturer::CapturedView& view);
    bool infer_view(ScreenCapturer::CapturedView& view);
    void mv(int mx, int my, double& ms);
    void mv_wind(int mx, int my, double& ms);
    void draw_ovl(double& ms);
    void log_lat(FrameLat& fl);
    void trt_log();
    // pure aim point for overlay, no state change
    cv::Point2f aim_no_pred(const Detection& d) const;
    // map detection reference to index in ds (linear scan; ds small)
    int det_index(const Detection& d) const;

    // state
    Config c;
    ScreenCapturer cap;
    ObjDet od;
    PoseInfer pose;
    Mouse mouse;
    AimAlgo algo;
    HotkeyManager hk;
    const cv::Rect crop;
    const cv::Point ctr;
    double mfac = 1.0;
    std::vector<Detection> ds;
    OCSort ocs;
    int    locked_track_id = -1; // 当前锁定的 track（按键按下维持）
    SelectedTarget cur_sel;      // 当前选择结果（obs+trk）
    std::vector<std::vector<trtyolo::KeyPoint>> pose_kpts; // same size as ds when filled
    std::vector<bool> pose_ok;
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

    // overlay state
    bool lock_ = false;
    OverlayRect lock_rect_{0,0,0,0};
    cv::Point2f aim_pt_{0.f, 0.f};
    bool aim_ok_ = false;

    // cache trks for overlay draw
    struct CachedTrk { cv::Rect trk; int id; int cls; float score; };
    std::vector<CachedTrk> last_trks_;

    WindState wind_;

    // grant access to helper free functions defined in separate units
    friend void aim_draw_cross(Aim* self);
    friend void aim_draw_lock(Aim* self);

    // small control helpers to shrink .cpp
    void on_no_target();
    void on_single(const Detection& target, double dt, double& mou_ms);
    void on_smooth(const Detection& target, double dt, double& mou_ms);
};

// inline impls (keep tiny to reduce .cpp size; no logic change)
inline void Aim::on_no_target(){ algo.reset(); last_tp_valid_ = false; lock_ = false; aim_ok_ = false; }
inline void Aim::on_single(const Detection& target, double dt, double& mou_ms){ bool use_pred = c.aim_single_use_pred; auto [dx, dy] = pred_err(target, dt, use_pred); int mx = (int)(dx * mfac), my = (int)(dy * mfac); mv(mx, my, mou_ms); }
inline void Aim::on_smooth(const Detection& target, double dt, double& mou_ms){ double h = (double)target.box.height; set_seg(h); auto [ex, ey] = pred_err(target, dt, true); auto pr = algo.step(ex, ey, dt); int mx = (int)(pr.first * mfac), my = (int)(pr.second * mfac); if (!c.wind_enable) mv(mx, my, mou_ms); else mv_wind(mx, my, mou_ms); }
