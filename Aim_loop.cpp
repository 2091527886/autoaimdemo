#include "Aim.hpp"
#include "pch.hpp"
#include "sh.hpp"

#include "CtrlHandler.hpp"
#include "OCSort.hpp"
#include "target_selector.hpp"
#include "PoseOCSBridge.hpp"

void Aim::loop() {
  while (true) {
    if (g_exit_req.load(std::memory_order_relaxed)) {
      break;
    }
    auto loop_start_time = tn();

    FrameLat fl;

    // reset per-frame timing accumulators to avoid cross-frame accumulation
    tm.infer_ms = 0.0;
    tm.post_ms = 0.0;
    tm.trt_throughput.clear();
    tm.trt_cpu_lat.clear();
    tm.trt_gpu_lat.clear();

    ScreenCapturer::CapturedView view;
    if (!capture(fl, view)) {
      log_detections(false);
      continue;
    }

    if (!infer_view(view)) {
      log_detections(false);
      continue;
    }
    log_detections(true);
    double ovl_ms = 0.0;
    fl.det_ms = c.timing ? (tm.infer_ms + tm.post_ms) : 0.0;

    bool smooth_pressed = hk.SmoothPressed();
    double dt_frame = get_dt();

    auto tctl0 = (c.timing ? tn() : tp{});
    double ctl_cost_ms = 0.0;
    double mou_cost_ms = 0.0;
    {
      // 0) 将 Timing 帧间隔传给 OCSort KF
      ocs.set_dt(static_cast<float>(dt_frame));
      // 使用 Pose→Host→Device 桥接接入 OCSort（仅用于目标选择，不参与瞄准追踪）
      std::vector<OCSort::Trk> trks;
      if (c.ocs_enable) {
        (void)ocs_update_from_pose_host(ds, ocs, cap.w(), cap.h(), trks, 0);
      } else {
        trks.reserve(ds.size()); int fake_id = 0;
        for (const auto& d : ds){
          OCSort::Trk t{}; t.x1 = static_cast<float>(d.box.x); t.y1 = static_cast<float>(d.box.y); t.x2 = static_cast<float>(d.box.x + d.box.width); t.y2 = static_cast<float>(d.box.y + d.box.height);
          t.ox1=t.x1; t.oy1=t.y1; t.ox2=t.x2; t.oy2=t.y2; t.score=d.confidence; t.cls=d.class_id; t.id=fake_id++; t.valid=true; trks.push_back(t);
        }
      }

      // 更新 overlay 缓存的全部 track 框（用于 ovl_show_trk_all）
      last_trks_.clear();
      last_trks_.reserve(trks.size());
      for (const auto& t : trks){ if (!t.valid) continue; CachedTrk ct; ct.trk = cv::Rect((int)t.x1,(int)t.y1,(int)(t.x2-t.x1),(int)(t.y2-t.y1)); ct.id=t.id; ct.cls=t.cls; ct.score=t.score; last_trks_.push_back(ct); }

      // 2) 目标选择（联合 class_id 与 track_id，默认 center_win）
      // cv::Point ctr_abs(ctr.x + crop.x, ctr.y + crop.y); // unused
      bool want_lock = smooth_pressed;
      SelectedTarget sel;
      if (want_lock && locked_track_id >= 0){
        // 严格维持锁定：若当前锁定的 track 仍存在且类别允许，则直接选它
        for (const auto& t : trks){
          if (!t.valid) continue;
          if (t.id == locked_track_id && c.is_target(t.cls)){
            sel.track_id = t.id; sel.cls = t.cls; sel.score = t.score;
            sel.obs = cv::Rect((int)t.ox1,(int)t.oy1,(int)(t.ox2-t.ox1),(int)(t.oy2-t.oy1));
            sel.trk = cv::Rect((int)t.x1,(int)t.y1,(int)(t.x2-t.x1),(int)(t.y2-t.y1));
            break;
          }
        }
      }
      if (!sel.valid()){
        sel = select_target(c, trks, ctr);
      }
      cur_sel = sel;

      if (!sel.valid()) {
        on_no_target();
      } else {
        if (!smooth_pressed) { adrc.reset(); mv_residual_ = {0.0, 0.0}; locked_track_id = -1; }
        lock_ = smooth_pressed;
        if (lock_ && (locked_track_id<0 || locked_track_id==sel.track_id)) locked_track_id = sel.track_id;
        if (!lock_) locked_track_id = -1;

        // 叠加层用 track 框；鼠标与瞄准完全使用“原始检测框 obs”
        lock_rect_ = { (float)(crop.x + sel.obs.x), (float)(crop.y + sel.obs.y), (float)sel.obs.width, (float)sel.obs.height };
        
        // 优先几何近似匹配 ds 中的检测，避免严格相等导致姿态丢失
        auto geom_match = [&](const cv::Rect& a, const cv::Rect& b){
          auto ca = cv::Point2f((float)a.x + a.width*0.5f, (float)a.y + a.height*0.5f);
          auto cb = cv::Point2f((float)b.x + b.width*0.5f, (float)b.y + b.height*0.5f);
          float dc = std::hypot(ca.x-cb.x, ca.y-cb.y);
          float sa = (float)a.area(), sb = (float)b.area();
          float srel = (sa>1 && sb>1) ? std::abs(sa-sb) / std::max(sa,sb) : 0.0f;
          return (dc <= c.sel_geom_center_eps_px) && (srel <= c.sel_geom_area_rel_eps);
        };
        
        const Detection* pDet = nullptr;
        if (sel.ds_idx >= 0 && sel.ds_idx < (int)ds.size()){
          const auto& d = ds[sel.ds_idx];
          if (d.class_id == sel.cls && geom_match(d.box, sel.obs)) pDet = &d;
        }
        if (!pDet){
          for (int i = 0; i < (int)ds.size(); ++i){
            const auto& d = ds[i];
            if (d.class_id != sel.cls) continue;
            if (geom_match(d.box, sel.obs)){ pDet = &d; break; }
          }
        }
        Detection det_fallback; det_fallback.box = sel.obs; det_fallback.confidence = sel.score; det_fallback.class_id = sel.cls;
        const Detection& det_for_aim = pDet ? *pDet : det_fallback;
        auto ap = aim_no_pred(det_for_aim);
        // FOV gate: circle centered at virtual screen center; outside => treat as no target (no smooth)
        bool in_fov = true;
        if (c.fov && c.fov_px > 0) {
          in_fov = fov_ok(ap.x, ap.y);
        }
        if (in_fov) {
          aim_pt_.x = ap.x;
          aim_pt_.y = ap.y;
          aim_ok_ = true;
        } else {
          aim_ok_ = false;
          lock_ = false;
        }
        // dbg centers
        dbg_has_pts_ = true;
        // track center（叠加层绘制使用 trk 框）
        dbg_trk_pt_.x = (float)(sel.trk.x + sel.trk.width*0.5f);
        dbg_trk_pt_.y = (float)(sel.trk.y + sel.trk.height*0.5f);
        // control center（用于 pred_err/ADRC 的 obs 中心）
        dbg_ctrl_pt_.x = (float)(det_for_aim.box.x + det_for_aim.box.width*0.5f);
        dbg_ctrl_pt_.y = (float)(det_for_aim.box.y + det_for_aim.box.height*0.5f);

        if (in_fov && smooth_pressed) { on_smooth(det_for_aim, dt_frame, mou_cost_ms); }
      }
    }
    auto tctl1 = (c.timing ? tn() : tctl0);
    ctl_cost_ms = c.timing ? msd(tctl1 - tctl0).count() : 0.0;

    fl.mou_ms = mou_cost_ms;
    fl.ctl_ms = MX(0.0, ctl_cost_ms - mou_cost_ms);

    draw_ovl(ovl_ms);
    fl.ovl_ms = ovl_ms;

    auto loop_end_time = tn();
    double proc_ms = msd(loop_end_time - loop_start_time).count();
    if (c.fps_lock) {
      tm.loop_ms = fc.end();
    } else {
      tm.loop_ms = proc_ms;
    }
    if (c.timing) {
      double loop_e2e = std::max({tm.loop_ms, fl.cap_ms, fl.det_ms, fl.mou_ms, fl.ctl_ms, fl.ovl_ms});
      fl.loop_ms = loop_e2e;
    }

    // 帧计数推进（用于日志/统计）
    c.frame_idx++;
    fl.idx = c.frame_idx;

    log_lat(fl);
    if (c.timing)
      cap.reporter.drain();
    trt_log();

    if (g_exit_req.load(std::memory_order_relaxed)) {
      break;
    }
  }
}
