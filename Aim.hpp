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
#include "HumanCursor.hpp"
#include "AimDZ.hpp"

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
    HumanCursor human;
    AimDZ dz;
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
    // dbg centers
    bool dbg_has_pts_ = false;
    cv::Point2f dbg_trk_pt_{0.f,0.f};
    cv::Point2f dbg_ctrl_pt_{0.f,0.f};
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
    friend void aim_draw_dz(Aim* self);
    friend void aim_draw_dbg(Aim* self);

    // small control helpers to shrink .cpp
    void on_no_target();
    void on_smooth(const Detection& target, double dt, double& mou_ms);
};

// inline impls (keep tiny to reduce .cpp size; no logic change)
inline void Aim::on_no_target(){ adrc.reset(); mv_residual_ = {0.0, 0.0}; human.reset(); lock_ = false; aim_ok_ = false; dbg_has_pts_ = false; }
inline void Aim::on_smooth(const Detection& target, double dt, double& mou_ms){
  auto [ex, ey] = pred_err(target);
  // deadzone on error to suppress endpoint jitter
  if (c.dz){ auto __dz = dz.apply(ex, ey, true); ex = __dz.first; ey = __dz.second; }
  if (!c.adrc_enable && c.human_mode==0) return; // OffsetCurve 模式依赖 ADRC；AbsoluteCurve 可不依赖

  if (c.human_enable && c.human_mode==1){
    // AbsoluteCurve：优先消耗已存在的步长；若队列为空则生成从当前光标到 aim 点的整条曲线
    double hx=0.0, hy=0.0;
    if (!human.pop_step(hx, hy)){
      cv::Point2f ap = aim_no_pred(target); // 屏幕坐标
      if (c.human_queue_clear_policy==1) human.clear_steps();
      // 使用配置的默认时长窗口映射到 target_points（不 sleep，不改计时）
      int ms_min = std::max(1, c.human_abs_dur_ms_min);
      int ms_max = std::max(ms_min, c.human_abs_dur_ms_max);
      static thread_local std::mt19937 _g{std::random_device{}()};
      std::uniform_int_distribution<int> _ud(ms_min, ms_max);
      double dur_s = _ud(_g) / 1000.0;
      human.move_to(ap.x, ap.y, dur_s);
      (void)human.pop_step(hx, hy); // 生成后立即尝试取一步
    }
    if (hx!=0.0 || hy!=0.0) mv_precise(hx, hy, mou_ms);
    return;
  }

  // OffsetCurve：ADRC 每帧输出微分量 → 人类化子曲线 → 逐帧弹出
  if (!c.adrc_enable) return;
  auto pr = adrc.step(ex, ey, dt);
  double wx = pr.first * mfac, wy = pr.second * mfac;
  if (c.human_enable){
    // 近目标极小步长抑制（仅在 OffsetCurve 路径使用）
    double l1 = std::abs(wx) + std::abs(wy);
    if (!(c.human_min_step_eps_enable && l1 < c.human_min_step_eps)){
      human.move_by_offset(wx, wy);
      double hx=0.0, hy=0.0; if (human.pop_step(hx, hy)) { mv_precise(hx, hy, mou_ms); }
    } else {
      // 直接由 mv_precise 用残差去消化，不生成曲线
      mv_precise(wx, wy, mou_ms);
    }
  } else {
    mv_precise(wx, wy, mou_ms);
  }
}
