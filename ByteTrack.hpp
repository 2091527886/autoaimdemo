#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "Detection.hpp"
#include "Config.hpp"

namespace bt {

// Full linear Kalman filter with 8D state [x,y,a,h, vx,vy,va,vh]
class KF8 {
public:
    KF8();
    void init(const cv::Rect& box);
    void predict();
    void update(const cv::Rect& box);
    cv::Rect toRect() const; // from current state
    // Mahalanobis gating distance for a candidate measurement
    double maha(const cv::Rect& box) const;
private:
    // helpers
    cv::Matx<double,4,1> toZ(const cv::Rect& box) const;

    // state and matrices as double for stability
    cv::Matx<double,8,1> x;  // state
    cv::Matx<double,8,8> F;  // transition
    cv::Matx<double,8,8> Q;  // process noise
    cv::Matx<double,4,8> H;  // observation
    cv::Matx<double,4,4> R;  // observation noise
    cv::Matx<double,8,8> P;  // covariance
    bool inited = false;
};

struct TrkNode {
    int id = -1;
    int class_id = -1;
    float last_conf = 0.0f;
    int age = 0;                 // total age
    int hits = 0;                // successful matches
    int time_since_update = 0;   // frames since last match
    KF8 kf;
    cv::Rect last_box;           // last measured or predicted
    bool alive = true;
};

struct BTParams {
    float high_th = 0.6f;
    float low_th  = 0.1f;
    float iou_th  = 0.3f;
    int   max_age = 30;
    int   min_hits = 3;
    bool  cls_match = true; // require class match when associating
    double maha_th = 9.49;  // Mahalanobis threshold
    bool out_nms = true;
    float out_nms_iou = 0.7f;
};

class ByteTracker {
public:
    explicit ByteTracker(const Config& c);
    // Update with detections for one frame and output tracked boxes.
    // If cfg.trk_pred_move is true, outputs predicted boxes for tracks without detection this frame.
    void update(const std::vector<Detection>& dets, std::vector<Detection>& out);
private:
    BTParams p;
    int next_id = 1;
    std::vector<TrkNode> tracks;

    static double IoU(const cv::Rect& a, const cv::Rect& b);
    // Hungarian assignment on cost matrix (lower cost better). Returns pairs (row->col)
    static std::vector<int> Hungarian(const std::vector<std::vector<double>>& cost);
};

} // namespace bt
