#pragma once

#include <vector>
#include <memory>
#include "Detection.hpp"
#include "Reporting.hpp"

// forward declarations
class Config;
struct Timing;
namespace trtyolo { class DetectModel; }
#include <cuda_runtime_api.h>

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

public:
    struct DevView { const float* xyxy=nullptr; const float* scores=nullptr; const int* classes=nullptr; int num=0; };
    const DevView& get_last_dev() const { return last_dev_; }
    cudaStream_t get_stream() const;
private:
    std::unique_ptr<trtyolo::DetectModel> model_;
    metrics::TrtPerfReporting perf_;
    DevView last_dev_;
};
