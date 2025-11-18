#pragma once

#include <memory>
#include <vector>
#include "trtyolo_inc.hpp"
#include "Reporting.hpp"
#include "Detection.hpp"

class Config;
struct Timing;

// Thin wrapper for TensorRT-YOLO PoseModel
class PoseInfer {
public:
    explicit PoseInfer(const Config& cfg);
    PoseInfer(const PoseInfer&) = delete;
    PoseInfer& operator=(const PoseInfer&) = delete;
    ~PoseInfer();

    // Run pose on a cropped BGRA view for selected detections.
    // inputs are the full captured view buffer and detections in the same view coordinates (0..view_w/h)
    // For each index in sel_idx, will fill out_kpts[idx] with kpts relative to the view (not full screen),
    // i.e., (x + det.box.x, y + det.box.y). If failed, out_ok[idx]=false.

    // full-view mode: predict once for the whole view
    bool run_on_view(unsigned char* data,
                     int view_w,
                     int view_h,
                     size_t pitch,
                     std::vector<std::vector<trtyolo::KeyPoint>>& out_kpts,
                     std::vector<cv::Rect>& out_boxes,
                     std::vector<int>& out_classes,
                     std::vector<float>& out_scores,
                     const Config& cfg,
                     Timing& tm);

private:
    std::unique_ptr<trtyolo::PoseModel> model_;
    metrics::TrtPerfReporting perf_;
};
