#include "ocs_ocm.cuh"

namespace ocs {

__global__ void k_momentum(const float2* prev, const float2* curr, int T, float eps,
                           float2* mom, float* valid){
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=T) return;
    float2 p = prev[i], c = curr[i];
    float dx = c.y - p.y; // X diff
    float dy = c.x - p.x; // Y diff (to align with (Y,X) usage)
    float n = sqrtf(dx*dx + dy*dy);
    if (n > eps){ mom[i] = make_float2(dy/n, dx/n); valid[i] = 1.0f; }
    else { mom[i] = make_float2(0.f,0.f); valid[i] = 0.0f; }
}

void ocm_compute_momentum(const float2* prev, const float2* curr, int T, float eps,
                          float2* mom, float* valid,
                          cudaStream_t stream){
    int tb=256, gb=(T+tb-1)/tb; k_momentum<<<gb,tb,0,stream>>>(prev,curr,T,eps,mom,valid);
}

} // namespace ocs
