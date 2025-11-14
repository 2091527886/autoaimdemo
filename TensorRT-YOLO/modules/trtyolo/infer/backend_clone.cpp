#include "infer/backend.hpp"
#include "utils/common.hpp"

namespace trtyolo {

std::unique_ptr<TrtBackend> TrtBackend::clone() {
    auto clone_backend          = std::make_unique<TrtBackend>();
    clone_backend->infer_config = infer_config;

    cudaSetDevice(infer_config.device_id);
    CHECK(cudaStreamCreate(&clone_backend->stream));

    clone_backend->zero_copy_ = zero_copy_;
    clone_backend->manager_   = manager_->clone();

    clone_backend->getTensorInfo();
    clone_backend->initialize();

    if (!clone_backend->dynamic) clone_backend->captureCudaGraph();

    return clone_backend;
}

} // namespace trtyolo
