#pragma once
#include <cuda_runtime.h>

namespace ocs {

// Batched Kalman Filter (linear), state dim 7: [cx, cy, s, r, vx, vy, vs]
// observation dim 4: [cx, cy, s, r]
// x:  T x 7, row-major contiguous
// P:  T x 49 (7x7 full covariance, row-major per track)
// z:  T x 4, observation per track (if matched), undefined if matched_mask[i]==0
// matched_mask: T (1 matched, 0 unmatched)

void kf_predict(float* x, float* P, int T, float dt,
                float q_pos, float q_size, cudaStream_t stream);

void kf_update(float* x, float* P, const float* z, const int* matched_mask, int T,
               float r_pos, float r_size, cudaStream_t stream);

} // namespace ocs
