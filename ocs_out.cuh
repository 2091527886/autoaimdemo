#pragma once
#include <cuda_runtime.h>

namespace ocs {

static_assert(sizeof(float4)==16, "float4 must be 16 bytes");

struct OcsPackedTrk {
    float4 trk;   // (x1,y1,x2,y2)
    float4 obs;   // (x1,y1,x2,y2)
    float  score; // confidence from meta_score
    int    cls;   // class id
    int    id;    // track id
    int    valid; // 1/0
};

static_assert(sizeof(OcsPackedTrk) == (sizeof(float4)*2 + sizeof(float) + sizeof(int)*3), "OcsPackedTrk layout mismatch");

// Launch device kernel to build compact outputs from KF state and per-track meta
void build_outputs(const float* kfx,
                   float4* out_trk,
                   float4* out_obs,
                   float*  out_score,
                   int*    out_cls,
                   int*    out_id,
                   int*    out_valid,
                   const int*    meta_id,
                   const int*    meta_cls,
                   const float*  meta_score,
                   const int*    meta_tsu,
                   const int*    meta_hit,
                   const float4* meta_obs,
                   int min_hits,
                   int T,
                   cudaStream_t stream);

// AoS packed builder to reduce D2H copies to a single memcpy
void build_outputs_packed(const float* kfx, // kfx ignored when KF disabled (still required for API)
                          OcsPackedTrk* out_packed,
                          const int*    meta_id,
                          const int*    meta_cls,
                          const float*  meta_score,
                          const int*    meta_tsu,
                          const int*    meta_hit,
                          const float4* meta_obs,
                          int min_hits,
                          int T,
                          cudaStream_t stream);

// Validation kernel to preflight-read outputs inputs (no side effects)
void validate_outputs_args(const float* kfx,
                           const int*    meta_id,
                           const int*    meta_cls,
                           const float*  meta_score,
                           const int*    meta_tsu,
                           const int*    meta_hit,
                           const float4* meta_obs,
                           int T,
                           cudaStream_t stream);

} // namespace ocs
