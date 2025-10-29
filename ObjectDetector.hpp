#pragma once

#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include "Detection.hpp"
#include "metrics/TrtPerfReporting.hpp"

// forward declarations
class Config;
struct TimingDetails;
namespace trtyolo { class DetectModel; }

class ObjectDetector {
public:
    explicit ObjectDetector(const Config& cfg);
    ObjectDetector(const ObjectDetector&) = delete;
    ObjectDetector& operator=(const ObjectDetector&) = delete;
    ~ObjectDetector();

    void Detect(const cv::Mat& image,
                std::vector<Detection>& detections,
                const Config& cfg,
                TimingDetails& timings);

private:
    std::unique_ptr<trtyolo::DetectModel> detector_;
    metrics::TrtPerfReporting trt_reporter_;
};
