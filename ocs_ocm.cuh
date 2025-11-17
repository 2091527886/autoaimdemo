#pragma once
#include <cuda_runtime.h>

namespace ocs {

// Coordinate convention notes:
// prev/curr are centers stored as float2(cy, cx)
// momentum (mom) is unit direction (dy, dx) to align with (Y,X)

// Compute per-track momentum (unit direction) from prev and current centers on device.
// prev: T x float2 (cx, cy)
// curr: T x float2 (cx, cy)
// mom:  T x float2 (out, unit (dy, dx)) to align with existing VDC convention (Y,X)
// valid: T (1 if distance>eps, else 0)
void ocm_compute_momentum(const float2* prev, const float2* curr, int T, float eps,
                          float2* mom, float* valid,
                          cudaStream_t stream);

} // namespace ocs
