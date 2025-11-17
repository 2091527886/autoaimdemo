#pragma once

#include "api.hpp"

namespace trtyolo {

class BaseModel; // fwd

class TRTYOLOAPI InferOption {
public:
    InferOption();
    ~InferOption();

    void setDeviceId(int id);
    void enableCudaMem();
    void enableManagedMemory();
    void enablePerformanceReport();
    void enableExportGPUDet();
    void set_rb();
    void enableSwapRB(); // 保持对外 API（内部仍可调用 set_rb）
    void setBorderValue(float border_value);

    void setLetterboxBlockXY(int bx, int by);
    void setLetterboxLaunchBounds(bool enable, int min_blocks_per_sm = 2);
    void enableLetterboxOccupancy(bool enable);

    void setNormalizeParams(const std::vector<float>& mean, const std::vector<float>& std);
    void setInputDimensions(int width, int height);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    friend class trtyolo::BaseModel;
};

} // namespace trtyolo
