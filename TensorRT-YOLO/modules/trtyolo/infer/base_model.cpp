#include "infer/trtyolo.hpp"
#include "infer/detail/trtyolo_impl.hpp"

namespace trtyolo {

BaseModel::BaseModel()  = default;
BaseModel::~BaseModel() = default;

BaseModel::BaseModel(const std::string& trt_engine_file, const InferOption& infer_option)
    : impl_(std::make_unique<Impl>(trt_engine_file, infer_option)) {}

int BaseModel::batch() const { return impl_->batch(); }

std::tuple<std::string, std::string, std::string> BaseModel::performanceReport() {
    return impl_->performanceReport();
}

} // namespace trtyolo
