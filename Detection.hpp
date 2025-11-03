#pragma once

#include <opencv2/core.hpp> // for cv::Rect

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
    int track_id = -1;      // ByteTrack assigned id; -1 if none
    bool is_pred = false;   // true if this box is a pure Kalman prediction (no detection this frame)
};
