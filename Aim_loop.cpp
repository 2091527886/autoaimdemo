#include "Aim.hpp"
#include "pch.hpp"
#include "sh.hpp"

#include "CtrlHandler.hpp"
#include "OCSort.hpp"
#include "target_selector.hpp"

void Aim::loop() {
  while (true) {
    if (g_exit_req.load(std::memory_order_relaxed)) {
      break;
    }
    auto loop_start_time = tn();

    FrameLat fl;

    ScreenCapturer::CapturedView view;
    if (!capture(fl, view)) {
      continue;
    }

    if (!infer_view(view)) {
      continue;
    }
    double ovl_ms = 0.0;
    fl.det_ms = c.timing ? (tm.infer_ms + tm.post_ms) : 0.0;

    bool smooth_pressed = hk.SmoothPressed();
    double dt_frame = get_dt();

    auto tctl0 = (c.timing ? tn() : tp{});
    double ctl_cost_ms = 0.0;
    double mou_cost_ms = 0.0;
    {
      // 0) 将 Timing 帧间隔传给 OCSort KF
      ocs.set_dt((float)dt_frame);
      // 1) 使用设备侧检测结果视图，直接调用 OCSort::update_dev（真·零拷贝链路）
      std::vector<OCSort::Trk> trks;
      const auto& dv = od.get_last_dev();
      if (dv.num > 0 && dv.xyxy && dv.scores && dv.classes) {
        // 与 trtyolo 串联同一流，端到端零等待
        ocs.update_dev(dv.xyxy, dv.scores, dv.classes, dv.num, trks, cap.w(), cap.h(), od.get_stream());
      } else {
        // 若无设备视图（极端情况），不进行 CPU 路径回退，直接输出空轨迹
        trks.clear();
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
        if (!smooth_pressed) { algo.reset(); last_tp_valid_ = false; locked_track_id = -1; }
        lock_ = smooth_pressed;
        if (lock_ && (locked_track_id<0 || locked_track_id==sel.track_id)) locked_track_id = sel.track_id;
        if (!lock_) locked_track_id = -1;

        // 叠加层用 track 框；鼠标与瞄准完全使用“原始检测框 obs”
        lock_rect_ = { (float)(crop.x + sel.obs.x), (float)(crop.y + sel.obs.y), (float)sel.obs.width, (float)sel.obs.height };
        // 优先在本帧 ds 中找到与 sel.obs 对应的元素，确保 aim_no_pred 能利用 pose 关键点
        auto iou_rect = [](const cv::Rect& a, const cv::Rect& b)->double{
          int x1 = std::max(a.x, b.x);
          int y1 = std::max(a.y, b.y);
          int x2 = std::min(a.x + a.width,  b.x + b.width);
          int y2 = std::min(a.y + a.height, b.y + b.height);
          int w = std::max(0, x2 - x1);
          int h = std::max(0, y2 - y1);
          double inter = 1.0 * w * h;
          double uni = 1.0 * a.area() + 1.0 * b.area() - inter + 1e-6;
          return inter / uni;
        };
        const Detection* pDet = nullptr;
        int best_i = -1; double best_iou = -1.0;
        for (int i = 0; i < (int)ds.size(); ++i){
          const auto& d = ds[i];
          if (d.class_id != sel.cls) continue;
          double u = iou_rect(d.box, sel.obs);
          if (u > best_iou){ best_iou = u; best_i = i; }
        }
        if (best_i >= 0 && best_iou >= (double)c.pose_match_iou_thresh){
          pDet = &ds[best_i];
        }
        Detection det_fallback; det_fallback.box = sel.obs; det_fallback.confidence = sel.score; det_fallback.class_id = sel.cls;
        const Detection& det_for_aim = pDet ? *pDet : det_fallback;
        auto ap = aim_no_pred(det_for_aim); 
        aim_pt_.x = ap.x; 
        aim_pt_.y = ap.y; 
        aim_ok_ = true;

        if (smooth_pressed) { on_smooth(det_for_aim, dt_frame, mou_cost_ms); }
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

    log_lat(fl);
    if (c.timing)
      cap.reporter.drain();
    trt_log();

    if (g_exit_req.load(std::memory_order_relaxed)) {
      break;
    }
  }
}
