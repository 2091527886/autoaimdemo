#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <random>
#include <chrono>
#include "Config.hpp"
#include "Timing.hpp"
#include "Detection.hpp"
#include "Mouse.hpp"
#include "HotkeyManager.hpp"
#include "ScreenCapturer.hpp"
#include "Pose.hpp"
#include "OCSort.hpp"
#include "target_selector.hpp"
#include "Game_overlay.h"
#include "FpsCtl.hpp"
#include "AimADRC.hpp"

class Aim {
private:
    // wind removed
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
    std::pair<double,double> pred_err(const Detection& d);
    bool capture(FrameLat& fl, ScreenCapturer::CapturedView& view);
    bool infer_view(ScreenCapturer::CapturedView& view);
    void mv_precise(double wx, double wy, double& ms);
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
    PoseInfer pose;
    Mouse mouse;
    AimADRC adrc;
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


    // overlay state
    bool lock_ = false;
    OverlayRect lock_rect_{0,0,0,0};
    cv::Point2f aim_pt_{0.f, 0.f};
    bool aim_ok_ = false;

    // cache trks for overlay draw
    struct CachedTrk { cv::Rect trk; int id; int cls; float score; };
    std::vector<CachedTrk> last_trks_;

   struct PendingText { float x; float y; std::wstring text; OverlayColor col; uint16_t layer; int pri; int ord; };
   std::vector<PendingText> pending_texts_;

   cv::Point2d mv_residual_{0.0, 0.0};
   // 轻量调试统计：量化损失累计与事件次数（仅在 timing 打印）
   double mvq_loss_sum_ = 0.0;
   size_t mvq_events_ = 0;

    // grant access to helper free functions defined in separate units
    friend void aim_draw_cross(Aim* self);
    friend void aim_draw_lock(Aim* self);

    // small control helpers to shrink .cpp
    void on_no_target();
    void on_smooth(const Detection& target, double dt, double& mou_ms);
};

// inline impls (keep tiny to reduce .cpp size; no logic change)
inline void Aim::on_no_target(){ adrc.reset(); mv_residual_ = {0.0, 0.0}; lock_ = false; aim_ok_ = false; }
inline void Aim::on_smooth(const Detection& target, double dt, double& mou_ms){
  auto [ex, ey] = pred_err(target);
  if (!c.adrc_enable) return;
  auto pr = adrc.step(ex, ey, dt);
  double wx = pr.first * mfac, wy = pr.second * mfac;
  mv_precise(wx, wy, mou_ms);
}
