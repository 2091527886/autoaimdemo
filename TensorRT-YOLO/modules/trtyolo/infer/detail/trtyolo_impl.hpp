#pragma once

// Internal implementation header for trtyolo Impl classes and helpers.
// This header is used only inside the library build and is not installed.

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "infer/trtyolo.hpp"
#include "infer/backend.hpp"
#include "utils/common.hpp"

namespace trtyolo {

class InferOption::Impl {
public:
    InferConfig getInferConfig() const;
    void setDeviceId(int id);
    void enableCudaMem();
    void enableManagedMemory();
    void enablePerformanceReport();
    void setInputDimensions(int width, int height);
    void set_rb();
    void setBorderValue(float value);
    void setLetterboxBlockXY(int bx, int by);
    void setLetterboxLaunchBounds(bool enable, int min_blocks_per_sm);
    void enableLetterboxOccupancy(bool enable);
    void setNormalizeParams(const std::vector<float>& mean, const std::vector<float>& std);

private:
    InferConfig infer_config;
    friend class trtyolo::BaseModel::Impl;
};

class BaseModel::Impl {
public:
    Impl();
    ~Impl();
    Impl(const std::string& trt_engine_file, const InferOption& infer_option);

    std::unique_ptr<Impl> clone() const;

    std::tuple<std::string, std::string, std::string> performanceReport();
    size_t batch() const;

    // Post-processors
    ClassifyRes postProcessClassify(int idx);
    DetectRes   postProcessDetect(int idx);
    OBBRes      postProcessOBB(int idx);
    SegmentRes  postProcessSegment(int idx);
    PoseRes     postProcessPose(int idx);

    // Decorator template kept inline to ensure visibility across TUs
    template <typename Func, typename ReturnType>
    ReturnType withPerformanceReport(const std::vector<Image>& images, Func func) {
        if (backend_->infer_config.enable_performance_report) {
            total_request_ += (backend_->dynamic ? images.size() : backend_->max_shape.x);
            infer_cpu_trace_->start();
            infer_gpu_trace_->start();
        }

        backend_->infer(images);
        ReturnType result = func(images.size());

        if (backend_->infer_config.enable_performance_report) {
            infer_gpu_trace_->stop();
            infer_cpu_trace_->stop();
        }
        return result;
    }

private:
    std::unique_ptr<TrtBackend> backend_;
    unsigned long long          total_request_{0};
    std::unique_ptr<GpuTimer>   infer_gpu_trace_;
    std::unique_ptr<CpuTimer>   infer_cpu_trace_;
};

} // namespace trtyolo
