#include "infer/backend.hpp"

namespace trtyolo {

void TrtBackend::infer(const std::vector<Image>& inputs) {
    cudaSetDevice(infer_config.device_id);
    if (dynamic) {
        dynamicInfer(inputs);
    } else {
        staticInfer(inputs);
    }
}

} // namespace trtyolo
