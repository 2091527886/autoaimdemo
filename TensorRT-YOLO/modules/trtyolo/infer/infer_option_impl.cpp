#include "infer/detail/trtyolo_impl.hpp"
#include "infer/backend.hpp"
#include <cassert>
#include <vector_functions.hpp>

namespace trtyolo {

InferConfig InferOption::Impl::getInferConfig() const { return infer_config; }
void InferOption::Impl::setDeviceId(int id) { infer_config.device_id = id; }
void InferOption::Impl::enableCudaMem() { infer_config.cuda_mem = true; }
void InferOption::Impl::enableManagedMemory() { infer_config.enable_managed_memory = true; }
void InferOption::Impl::enablePerformanceReport() { infer_config.enable_performance_report = true; }
void InferOption::Impl::enableExportGPUDet() { infer_config.export_gpu_det = true; }
void InferOption::Impl::setInputDimensions(int width, int height) { infer_config.input_shape = make_int2(height, width); }
void InferOption::Impl::set_rb() { infer_config.config.rb = true; }
void InferOption::Impl::setBorderValue(float value) { infer_config.config.border_value = value; }
void InferOption::Impl::setLetterboxBlockXY(int bx, int by) {
    infer_config.config.block_x = bx;
    infer_config.config.block_y = by;
}
void InferOption::Impl::setLetterboxLaunchBounds(bool enable, int min_blocks_per_sm) {
    infer_config.config.use_launch_bounds = enable;
    infer_config.config.min_blocks_per_sm = min_blocks_per_sm;
}
void InferOption::Impl::enableLetterboxOccupancy(bool enable) {
    infer_config.config.use_occupancy = enable;
}
void InferOption::Impl::setNormalizeParams(const std::vector<float>& mean, const std::vector<float>& std) {
    assert(mean.size() == 3 && std.size() == 3 && "ProcessConfig: requires the size of mean and std to be 3.");
    infer_config.config.alpha.x = 1.0 / 255.0f / std[0];
    infer_config.config.alpha.y = 1.0 / 255.0f / std[1];
    infer_config.config.alpha.z = 1.0 / 255.0f / std[2];
    infer_config.config.beta.x  = -mean[0] / std[0];
    infer_config.config.beta.y  = -mean[1] / std[1];
    infer_config.config.beta.z  = -mean[2] / std[2];
}

} // namespace trtyolo
