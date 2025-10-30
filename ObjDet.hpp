#pragma once

#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include "Detection.hpp"
#include "Reporting.hpp"

// forward declarations
class Config;
struct TimingDetails;
namespace trtyolo { class DetectModel; }

class ObjDet {
public:
    explicit ObjDet(const Config& cfg);
    ObjDet(const ObjDet&) = delete;
    ObjDet& operator=(const ObjDet&) = delete;
    ~ObjDet();

    void detect(const cv::Mat& img,
                std::vector<Detection>& out,
                const Config& cfg,
                TimingDetails& tms);

private:
    std::unique_ptr<trtyolo::DetectModel> model_;
    metrics::TrtPerfReporting perf_;
};
