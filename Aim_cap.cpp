#include "Aim.hpp"
#include "sh.hpp"
#include <algorithm>
#include "objdet_helpers.hpp"

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
  c.frame_idx++;
  fl.idx = c.frame_idx;
  fl.cap_ms = cap_ms;
  return true;
}

bool Aim::infer_view(ScreenCapturer::CapturedView &view) {
  if (view.mapped && view.data) {
    od.detect_view(view.data, view.width, view.height, view.pitch, ds, c, tm);
    // run pose
    pose_kpts.clear(); pose_ok.clear();
    if (c.pose_enable) {
      if (c.pose_full_view) {
        std::vector<std::vector<trtyolo::KeyPoint>> all_kpts;
        std::vector<cv::Rect> all_boxes;
        if (pose.run_on_view(view.data, view.width, view.height, view.pitch, all_kpts, all_boxes, c, tm)){
          pose_kpts.assign(ds.size(), {});
          pose_ok.assign(ds.size(), false);
          std::vector<int> used(all_boxes.size(), -1);
          for (size_t i=0;i<ds.size();++i){
            if (!c.is_target(ds[i].class_id)) continue;
            int best=-1; float best_iou=0.f;
            for (size_t j=0;j<all_boxes.size();++j){ float iou = box_iou(ds[i].box, all_boxes[j]); if (iou>best_iou){ best_iou=iou; best=(int)j; } }
            if (best>=0 && best_iou >= c.pose_match_iou_thresh && used[best]==-1){ pose_kpts[i]=all_kpts[best]; pose_ok[i]=true; used[best]=(int)i; }
          }
        }
      } else {
        // crop-topK mode (previous)
        std::vector<int> idx(ds.size()); for (size_t i=0;i<ds.size();++i) idx[i]=(int)i;
        std::stable_sort(idx.begin(), idx.end(), [&](int a,int b){
          if (!c.pose_prior_center) return ds[a].confidence>ds[b].confidence;
          auto ca = cv::Point(ds[a].box.x + ds[a].box.width/2, ds[a].box.y + ds[a].box.height/2);
          auto cb = cv::Point(ds[b].box.x + ds[b].box.width/2, ds[b].box.y + ds[b].box.height/2);
          double da = HYP(ca.x-ctr.x, ca.y-ctr.y), db = HYP(cb.x-ctr.x, cb.y-ctr.y);
          double ba = (c.pose_center_radius_px>0 && da <= c.pose_center_radius_px) ? 0.1 : 0.0;
          double bb = (c.pose_center_radius_px>0 && db <= c.pose_center_radius_px) ? 0.1 : 0.0;
          double sa = ds[a].confidence + ba - c.pose_center_lambda * da;
          double sb = ds[b].confidence + bb - c.pose_center_lambda * db;
          return sa>sb;
        });
        std::vector<int> sel; sel.reserve((size_t)MN((int)idx.size(), c.pose_run_topk));
        for (int id : idx) { if ((int)sel.size()>=c.pose_run_topk) break; if (!c.is_target(ds[id].class_id)) continue; if (ds[id].box.width < c.pose_min_box_px || ds[id].box.height < c.pose_min_box_px) continue; sel.push_back(id); }
        pose.run_on_crops(view.data, view.width, view.height, view.pitch, ds, sel, pose_kpts, pose_ok, c, tm);
      }
    }

    cap.release_view(view);
    return true;
  }
  cap.release_view(view);
  return false;
}
