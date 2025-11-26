#include "ocs_out.cuh"

namespace ocs {

__global__ void k_build_outputs_dev(const float* kfx,
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
                                    int T){
    int j = blockIdx.x*blockDim.x + threadIdx.x; if (j>=T) return;
    float cx=kfx[j*7+0], cy=kfx[j*7+1], s=kfx[j*7+2], r=kfx[j*7+3];
    float w = sqrtf(fabsf(s)*fabsf(r));
    float h = 0.f;
    if (w > 1e-3f) h = s / w;
    float4 tb = make_float4(cx-0.5f*w, cy-0.5f*h, cx+0.5f*w, cy+0.5f*h);
    out_trk[j] = tb;
    out_obs[j] = meta_obs[j];
    out_score[j] = meta_score[j];
    out_cls[j]   = meta_cls[j];
    out_id[j]    = meta_id[j];
    int v = (meta_tsu[j]==0 && meta_hit[j] >= min_hits) ? 1 : 0;
    out_valid[j] = v;
}

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
                   cudaStream_t stream){
    int tb=256, gb=(T+tb-1)/tb;
    k_build_outputs_dev<<<gb,tb,0,stream>>>(kfx, out_trk, out_obs, out_score, out_cls, out_id, out_valid,
                                            meta_id, meta_cls, meta_score, meta_tsu, meta_hit, meta_obs,
                                            min_hits, T);
}

__global__ void k_build_outputs_packed(const float* kfx,
                                         OcsPackedTrk* out_packed,
                                         const int*    meta_id,
                                         const int*    meta_cls,
                                         const float*  meta_score,
                                         const int*    meta_tsu,
                                         const int*    meta_hit,
                                         const float4* meta_obs,
                                         int min_hits,
                                         int T){
    int j = blockIdx.x*blockDim.x + threadIdx.x; if (j>=T) return;
    float cx=kfx[j*7+0], cy=kfx[j*7+1], s=kfx[j*7+2], r=kfx[j*7+3];
    float w = sqrtf(fabsf(s)*fabsf(r));
    float h = 0.f; if (w>1e-3f) h = s / w;
    float4 tb = make_float4(cx-0.5f*w, cy-0.5f*h, cx+0.5f*w, cy+0.5f*h);
    OcsPackedTrk out;
    out.trk = tb;
    out.obs = meta_obs[j];
    out.score = meta_score[j];
    out.cls   = meta_cls[j];
    out.id    = meta_id[j];
    out.valid = (meta_tsu[j]==0 && meta_hit[j] >= min_hits) ? 1 : 0;
    out_packed[j] = out;
}

void build_outputs_packed(const float* kfx,
                          OcsPackedTrk* out_packed,
                          const int*    meta_id,
                          const int*    meta_cls,
                          const float*  meta_score,
                          const int*    meta_tsu,
                          const int*    meta_hit,
                          const float4* meta_obs,
                          int min_hits,
                          int T,
                          cudaStream_t stream){
    int tb=256, gb=(T+tb-1)/tb;
    k_build_outputs_packed<<<gb,tb,0,stream>>>(kfx, out_packed, meta_id, meta_cls, meta_score, meta_tsu, meta_hit, meta_obs, min_hits, T);
}


__global__ void k_validate_outputs(const float* kfx,
                                   const int*    meta_id,
                                   const int*    meta_cls,
                                   const float*  meta_score,
                                   const int*    meta_tsu,
                                   const int*    meta_hit,
                                   const float4* meta_obs,
                                   int T){
    int j = blockIdx.x*blockDim.x + threadIdx.x; if (j>=T) return;
    // read a few elements to force memory access; no write
    volatile float a = 0.f;
    a += kfx ? kfx[j*7+0] : 0.f;
    a += meta_id ? (float)meta_id[j] : 0.f;
    a += meta_cls ? (float)meta_cls[j] : 0.f;
    a += meta_score ? meta_score[j] : 0.f;
    a += meta_tsu ? (float)meta_tsu[j] : 0.f;
    a += meta_hit ? (float)meta_hit[j] : 0.f;
    if (meta_obs){ float4 b = meta_obs[j]; a += b.x + b.y + b.z + b.w; }
}

void validate_outputs_args(const float* kfx,
                           const int*    meta_id,
                           const int*    meta_cls,
                           const float*  meta_score,
                           const int*    meta_tsu,
                           const int*    meta_hit,
                           const float4* meta_obs,
                           int T,
                           cudaStream_t stream){
    if (T<=0) return; int tb=256, gb=(T+tb-1)/tb; k_validate_outputs<<<gb,tb,0,stream>>>(kfx,meta_id,meta_cls,meta_score,meta_tsu,meta_hit,meta_obs,T);
}

} // namespace ocs
