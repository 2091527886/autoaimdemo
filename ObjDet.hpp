#pragma once

#include <vector>
#include <memory>
#include "Detection.hpp"
#include "Reporting.hpp"

// forward declarations
class Config;
struct Timing;
namespace trtyolo { class DetectModel; }

class ObjDet {
public:
    explicit ObjDet(const Config& cfg);
    ObjDet(const ObjDet&) = delete;
    ObjDet& operator=(const ObjDet&) = delete;
    ~ObjDet();

    // Zero-copy detect from BGRA pointer (pitch in bytes)
    void detect_view(unsigned char* data,
                     int width,
                     int height,
                     size_t pitch,
                     std::vector<Detection>& out,
                     const Config& cfg,
                     Timing& tm);

private:
    std::unique_ptr<trtyolo::DetectModel> model_;
    metrics::TrtPerfReporting perf_;
};
