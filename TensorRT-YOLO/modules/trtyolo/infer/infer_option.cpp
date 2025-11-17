#include "infer/trtyolo.hpp"
#include "infer/detail/trtyolo_impl.hpp"

namespace trtyolo {

InferOption::InferOption() : impl_(std::make_unique<InferOption::Impl>()) {}
InferOption::~InferOption() = default;

void InferOption::setDeviceId(int id) { impl_->setDeviceId(id); }
void InferOption::enableCudaMem() { impl_->enableCudaMem(); }
void InferOption::enableManagedMemory() { impl_->enableManagedMemory(); }
void InferOption::enablePerformanceReport() { impl_->enablePerformanceReport(); }
void InferOption::set_rb() { impl_->set_rb(); }
void InferOption::enableSwapRB() { impl_->set_rb(); }
void InferOption::setBorderValue(float border_value) { impl_->setBorderValue(border_value); }
void InferOption::setLetterboxBlockXY(int bx, int by) { impl_->setLetterboxBlockXY(bx, by); }
void InferOption::setLetterboxLaunchBounds(bool enable, int min_blocks_per_sm) { impl_->setLetterboxLaunchBounds(enable, min_blocks_per_sm); }
void InferOption::enableLetterboxOccupancy(bool enable) { impl_->enableLetterboxOccupancy(enable); }
void InferOption::setNormalizeParams(const std::vector<float>& mean, const std::vector<float>& std) { impl_->setNormalizeParams(mean, std); }
void InferOption::setInputDimensions(int width, int height) { impl_->setInputDimensions(height, width); }
void InferOption::enableExportGPUDet() { impl_->enableExportGPUDet(); }

} // namespace trtyolo
