// Split implementation moved into multiple files under infer/*.cpp
// This TU is intentionally left minimal to avoid duplicate definitions.

#include "infer/detail/trtyolo_impl.hpp"
#include <cuda_runtime.h>

namespace trtyolo {

// Ensure Impl frees per-batch dev buffers
BaseModel::Impl::~Impl(){
    for (auto* p : dev_boxes_per_batch_) { if (p) cudaFree(p); }
    dev_boxes_per_batch_.clear();
    dev_boxes_capacity_bytes_.clear();
}

}