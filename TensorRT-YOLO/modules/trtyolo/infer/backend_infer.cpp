#include "infer/backend.hpp"

namespace trtyolo {

void TrtBackend::infer(const std::vector<Image>& inputs) {
    // One-time pitch sanity check for BGRA expectation
    static bool s_pitch_checked = false;
    if (!s_pitch_checked && !inputs.empty()) {
        size_t expected = static_cast<size_t>(inputs.front().width) * 4;
        if (inputs.front().pitch != expected) {
            std::cerr << "[TRTYOLO] Warning: Image pitch (" << inputs.front().pitch
                      << ") != width*4 (" << expected
                      << ") while kernels expect BGRA. If source has padding, ensure passing correct pitch."
                      << " cuda_mem=" << (infer_config.cuda_mem ? "true" : "false")
                      << std::endl;
        }
        s_pitch_checked = true;
    }

    cudaSetDevice(infer_config.device_id);
    if (dynamic) {
        dynamicInfer(inputs);
    } else {
        staticInfer(inputs);
    }
}

} // namespace trtyolo
