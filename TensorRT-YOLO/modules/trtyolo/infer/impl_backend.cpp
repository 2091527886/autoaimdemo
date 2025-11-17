#include "infer/detail/trtyolo_impl.hpp"
#include "infer/backend.hpp"
#include "utils/common.hpp"
#include <sstream>
#include <vector>

namespace trtyolo {

BaseModel::Impl::Impl() = default;
// dtor is defined in trtyolo.cpp to free per-batch device buffers
// BaseModel::Impl::~Impl() is defined elsewhere

BaseModel::Impl::Impl(const std::string& trt_engine_file, const InferOption& infer_option)
    : backend_(std::make_unique<TrtBackend>(trt_engine_file, infer_option.impl_->getInferConfig())) {
    if (backend()->infer_config.enable_performance_report) {
        infer_gpu_trace_ = std::make_unique<GpuTimer>(backend()->stream);
        infer_cpu_trace_ = std::make_unique<CpuTimer>();
    }
    return;
}

std::unique_ptr<BaseModel::Impl> BaseModel::Impl::clone() const {
    auto clone_impl              = std::make_unique<Impl>();
    clone_impl->backend_         = backend_->clone();
    clone_impl->infer_gpu_trace_ = std::make_unique<GpuTimer>(clone_impl->backend_->stream);
    clone_impl->infer_cpu_trace_ = std::make_unique<CpuTimer>();
    return clone_impl;
}

std::tuple<std::string, std::string, std::string> BaseModel::Impl::performanceReport() {
    if (backend_->infer_config.enable_performance_report) {
        float throughput = total_request_ / infer_cpu_trace_->totalMilliseconds() * 1000;
        std::stringstream ss;

        ss << "Throughput: " << throughput << " qps";
        std::string throughputStr = ss.str();
        ss.str("");

        auto percentiles = std::vector<float>{90, 95, 99};
        auto getLatencyStr = [&](const auto& trace, const std::string& device) {
            auto result = getPerformanceResult(trace->milliseconds(), {0.90, 0.95, 0.99});
            ss << device << " Latency: min = " << result.min << " ms, max = " << result.max
               << " ms, mean = " << result.mean << " ms, median = " << result.median << " ms";
            for (int32_t i = 0, n = static_cast<int32_t>(percentiles.size()); i < n; ++i) {
                ss << ", percentile(" << percentiles[i] << "%) = " << result.percentiles[i] << " ms";
            }
            std::string output = ss.str();
            ss.str("");
            return output;
        };

        std::string cpuLatencyStr = getLatencyStr(infer_cpu_trace_, "CPU");
        std::string gpuLatencyStr = getLatencyStr(infer_gpu_trace_, "GPU");

        total_request_ = 0;
        infer_cpu_trace_->reset();
        infer_gpu_trace_->reset();

        return std::make_tuple(throughputStr, cpuLatencyStr, gpuLatencyStr);
    }
    return std::make_tuple("", "", "");
}

size_t BaseModel::Impl::batch() const { return backend_->max_shape.x; }

} // namespace trtyolo
