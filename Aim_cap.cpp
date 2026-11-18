#include "Aim.hpp"
#include "sh.hpp"
#include <algorithm>

bool Aim::capture(FrameLat &fl, ScreenCapturer::CapturedView &view) {
  double cap_ms = 0.0; bool got = false;
  if (c.timing) {
    auto t0 = tn();
    got = cap.capture_view(view, crop);
    auto t1 = tn();
    cap_ms = msd(t1 - t0).count();
  } else {
    got = cap.capture_view(view, crop);
  }
  if (!got) {
    fl.cap_ms = cap_ms;
    if (c.timing && c.log_latency && !c.latency_on_capture_only) {
      lat.push(fl);
    }
    return false;
  }
  
  fl.idx = c.frame_idx;
  fl.cap_ms = cap_ms;
  return true;
}

bool Aim::infer_view(ScreenCapturer::CapturedView &view) {
  if (view.mapped && view.data) {
    // 使用 pose 作为唯一检测来源（mpose.engine）
    pose_kpts.clear();
    pose_ok.clear();
    std::vector<std::vector<trtyolo::KeyPoint>> all_kpts;
    std::vector<cv::Rect> all_boxes;
    std::vector<int> all_classes;
    std::vector<float> all_scores;
    if (c.pose_enable && pose.run_on_view(view.data, view.width, view.height, view.pitch, all_kpts, all_boxes, all_classes, all_scores, c, tm)) {
      // build ds 1:1 from pose results
      ds.clear();
      ds.reserve(all_boxes.size());
      pose_kpts.reserve(all_boxes.size());
      pose_ok.reserve(all_boxes.size());
      for (size_t i=0;i<all_boxes.size();++i){
        if (i < all_scores.size() && all_scores[i] < c.conf_thresh) continue;
        Detection d; d.box = all_boxes[i]; d.class_id = (i<(size_t)all_classes.size()? all_classes[i] : 0); d.confidence = (i<(size_t)all_scores.size()? all_scores[i] : 0.f);
        if (d.box.width<=0 || d.box.height<=0) continue;
        ds.emplace_back(d);
        pose_kpts.emplace_back(i<all_kpts.size()? all_kpts[i] : std::vector<trtyolo::KeyPoint>{});
        pose_ok.push_back(!pose_kpts.back().empty());
      }
    } else {
      ds.clear();
    }

    cap.release_view(view);
    return true;
  }
  cap.release_view(view);
  return false;
}
