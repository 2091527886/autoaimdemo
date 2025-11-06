#pragma once

#include <opencv2/core.hpp> // for cv::Rect

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};
