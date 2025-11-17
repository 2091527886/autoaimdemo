#pragma once

#include "api.hpp"
#include "base_model.hpp"
#include "types.hpp"
#include "results.hpp"
#include <cuda_runtime_api.h>

namespace trtyolo {

class TRTYOLOAPI DetectModel : public BaseModel {
public:
    DetectModel();
    ~DetectModel();

    explicit DetectModel(const std::string& trt_engine_file, const InferOption& infer_option);

    std::unique_ptr<DetectModel> clone() const;

    DetectRes predict(const Image& image);
    std::vector<DetectRes> predict(const std::vector<Image>& images);

    // 设备侧输出（零拷贝）：仅在 InferOption 打开 GPU 导出时可用
    DetectResDev predict_dev(const Image& image);
    std::vector<DetectResDev> predict_dev(const std::vector<Image>& images);

    // 暴露底层 CUDA 流，便于与 OCSort 串联
    cudaStream_t getStream() const;
};

} // namespace trtyolo
