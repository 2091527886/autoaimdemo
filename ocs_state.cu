#include "ocs_state.cuh"
#include <cuda_runtime.h>
#include <math.h>

namespace ocs {

__device__ int g_append_overflow_count = 0;
__device__ int g_append_max_index = -1;

__global__ void k_reset_append_debug(){ g_append_overflow_count = 0; g_append_max_index = -1; }
void reset_append_debug(cudaStream_t stream){ k_reset_append_debug<<<1,1,0,stream>>>(); }
void read_append_debug(int* overflow_count, int* max_index){
    if (overflow_count) cudaMemcpyFromSymbol(overflow_count, g_append_overflow_count, sizeof(int));
    if (max_index) cudaMemcpyFromSymbol(max_index, g_append_max_index, sizeof(int));
}

__global__ void k_kfx_to_boxes(const float* kfx, int T, Box* out_boxes){
    int j = blockIdx.x*blockDim.x + threadIdx.x; if (j>=T) return;
    float cx=kfx[j*7+0], cy=kfx[j*7+1], s=kfx[j*7+2], r=kfx[j*7+3];
    float w = sqrtf(fabsf(s)*fabsf(r));
    float h = 0.f; if (w > 1e-3f) h = s / w;
    Box b; b.x1=cx-0.5f*w; b.y1=cy-0.5f*h; b.x2=cx+0.5f*w; b.y2=cy+0.5f*h; out_boxes[j]=b;
}

void kfx_to_boxes(const float* kfx, int T, Box* out_boxes, cudaStream_t stream){
    int tb=256, gb=(T+tb-1)/tb;
    k_kfx_to_boxes<<<gb,tb,0,stream>>>(kfx, T, out_boxes);
}

__global__ void k_obs_to_boxes(const float4* meta_obs, int T, Box* out_boxes){
    int j = blockIdx.x*blockDim.x + threadIdx.x; if (j>=T) return;
    float4 ob = meta_obs[j];
    Box b; b.x1=ob.x; b.y1=ob.y; b.x2=ob.z; b.y2=ob.w; out_boxes[j]=b;
}
void obs_to_boxes(const float4* meta_obs, int T, Box* out_boxes, cudaStream_t stream){
    int tb=256, gb=(T+tb-1)/tb;
    k_obs_to_boxes<<<gb,tb,0,stream>>>(meta_obs, T, out_boxes);
}

__global__ void k_age_tsu_begin(int T, int* meta_tsu){ int j=blockIdx.x*blockDim.x+threadIdx.x; if (j<T) meta_tsu[j]+=1; }
void age_tsu_begin(int T, int* meta_tsu, cudaStream_t stream){ int tb=256, gb=(T+tb-1)/tb; k_age_tsu_begin<<<gb,tb,0,stream>>>(T, meta_tsu); }

__global__ void k_update_hit_tsu(const int* sel_det, int T, int* meta_hit, int* meta_tsu){ int j=blockIdx.x*blockDim.x+threadIdx.x; if (j>=T) return; if (sel_det[j] >= 0){ meta_tsu[j]=0; meta_hit[j]+=1; }}
void update_hit_tsu_after_match(const int* sel_det, int T, int* meta_hit, int* meta_tsu, cudaStream_t stream){ int tb=256, gb=(T+tb-1)/tb; k_update_hit_tsu<<<gb,tb,0,stream>>>(sel_det, T, meta_hit, meta_tsu); }

__global__ void k_append_new_tracks(
    const float* det_xyxy, const float* det_scores, const int* det_cls,
    const int* unused_idx, int K,
    int D,
    int cap_M,
    int* trk_count, int* next_id,
    float* kf_x, float* kf_P,
    int* meta_id, int* meta_cls, float* meta_score, int* meta_tsu, int* meta_hit,
    float4* meta_obs,
    float2* obs_prev, float2* obs_curr,
    int* trk_cls
){
    int u = blockIdx.x*blockDim.x + threadIdx.x; if (u>=K) return;
    int di = unused_idx[u];
    if (di < 0 || di >= D) return;
    int idx = atomicAdd(trk_count, 1);
    if (idx >= cap_M) { atomicAdd(&g_append_overflow_count, 1); atomicSub(trk_count, 1); return; } // overflow guard with rollback
    atomicMax(&g_append_max_index, idx);
    int id = atomicAdd(next_id, 1);
    // det box
    float x1=det_xyxy[di*4+0], y1=det_xyxy[di*4+1], x2=det_xyxy[di*4+2], y2=det_xyxy[di*4+3];
    float w = fmaxf(1e-3f, x2-x1), h=fmaxf(1e-3f, y2-y1);
    float cx = x1 + 0.5f*w, cy=y1+0.5f*h, s=w*h, r=w/h;
    int off = idx*7, offP=idx*49;
    kf_x[off+0]=cx; kf_x[off+1]=cy; kf_x[off+2]=s; kf_x[off+3]=r; kf_x[off+4]=0; kf_x[off+5]=0; kf_x[off+6]=0;
    // P 初始化对角
    for (int rr=0; rr<7; ++rr){ for (int cc=0; cc<7; ++cc){ kf_P[offP + rr*7 + cc] = 0.0f; } }
    kf_P[offP + 0*7+0]=1.0f; kf_P[offP + 1*7+1]=1.0f; kf_P[offP + 2*7+2]=10.0f; kf_P[offP + 3*7+3]=10.0f; kf_P[offP + 4*7+4]=1.0f; kf_P[offP + 5*7+5]=1.0f; kf_P[offP + 6*7+6]=10.0f;
    // meta
    meta_id[idx]=id; meta_cls[idx]=det_cls[di]; meta_score[idx]=det_scores[di]; meta_tsu[idx]=0; meta_hit[idx]=1;
    meta_obs[idx]=make_float4(x1,y1,x2,y2);
    float2 c = make_float2(y1+y2, x1+x2); c.x*=0.5f; c.y*=0.5f; obs_prev[idx]=c; obs_curr[idx]=c;
    trk_cls[idx]=det_cls[di];
}

void append_new_tracks(const float* det_xyxy, const float* det_scores, const int* det_cls,
                       const int* unused_idx, int K,
                       int D,
                       int cap_M,
                       int* trk_count, int* next_id,
                       float* kf_x, float* kf_P,
                       int* meta_id, int* meta_cls, float* meta_score, int* meta_tsu, int* meta_hit,
                       float4* meta_obs,
                       float2* obs_prev, float2* obs_curr,
                       int* trk_cls,
                       cudaStream_t stream){
    int tb=256, gb=(K+tb-1)/tb; if (K>0) k_append_new_tracks<<<gb,tb,0,stream>>>(det_xyxy, det_scores, det_cls, unused_idx, K, D, cap_M, trk_count, next_id, kf_x, kf_P, meta_id, meta_cls, meta_score, meta_tsu, meta_hit, meta_obs, obs_prev, obs_curr, trk_cls);
}

__global__ void k_validate_append(const float* det_xyxy, const float* det_scores, const int* det_cls,
                                  const int* unused_idx, int K, int D,
                                  int cap_M,
                                  const float* kf_x, const float* kf_P,
                                  const int* meta_id, const int* meta_cls, const float* meta_score, const int* meta_tsu, const int* meta_hit,
                                  const float4* meta_obs,
                                  const float2* obs_prev, const float2* obs_curr,
                                  const int* trk_cls){
    int u = blockIdx.x*blockDim.x + threadIdx.x;
    if (u < K){
        int di = unused_idx ? unused_idx[u] : -1;
        if (di >= 0 && di < D){
            volatile float s = 0.f;
            if (det_xyxy){ s += det_xyxy[di*4+0]; }
            if (det_scores){ s += det_scores[di]; }
            if (det_cls){ s += (float)det_cls[di]; }
        }
    }
    if (u == 0){
        int last = cap_M > 0 ? cap_M-1 : 0;
        volatile float s = 0.f;
        if (kf_x){ s += kf_x[0]; if (cap_M>0) s += kf_x[last*7]; }
        if (kf_P){ s += kf_P[0]; if (cap_M>0) s += kf_P[last*49]; }
        if (meta_id){ s += (float)meta_id[0]; if (cap_M>0) s += (float)meta_id[last]; }
        if (meta_cls){ s += (float)meta_cls[0]; if (cap_M>0) s += (float)meta_cls[last]; }
        if (meta_score){ s += meta_score[0]; if (cap_M>0) s += meta_score[last]; }
        if (meta_tsu){ s += (float)meta_tsu[0]; if (cap_M>0) s += (float)meta_tsu[last]; }
        if (meta_hit){ s += (float)meta_hit[0]; if (cap_M>0) s += (float)meta_hit[last]; }
        if (meta_obs){ float4 a = meta_obs[0]; s += a.x; if (cap_M>0){ float4 b = meta_obs[last]; s += b.x; } }
        if (obs_prev){ float2 a = obs_prev[0]; s += a.x; if (cap_M>0){ float2 b = obs_prev[last]; s += b.x; } }
        if (obs_curr){ float2 a = obs_curr[0]; s += a.x; if (cap_M>0){ float2 b = obs_curr[last]; s += b.x; } }
        if (trk_cls){ s += (float)trk_cls[0]; if (cap_M>0) s += (float)trk_cls[last]; }
    }
}

void validate_append_args(const float* det_xyxy, const float* det_scores, const int* det_cls,
                          const int* unused_idx, int K, int D,
                          int cap_M,
                          const float* kf_x, const float* kf_P,
                          const int* meta_id, const int* meta_cls, const float* meta_score, const int* meta_tsu, const int* meta_hit,
                          const float4* meta_obs,
                          const float2* obs_prev, const float2* obs_curr,
                          const int* trk_cls,
                          cudaStream_t stream){
    int tb=256, gb=(max(K,1)+tb-1)/tb; k_validate_append<<<gb,tb,0,stream>>>(det_xyxy, det_scores, det_cls, unused_idx, K, D, cap_M, kf_x, kf_P, meta_id, meta_cls, meta_score, meta_tsu, meta_hit, meta_obs, obs_prev, obs_curr, trk_cls);
}

__global__ void k_validate_unused_idx(const int* unused_idx, int K, int D,
                                      int* bad_count, int* first_bad_u, int* first_bad_val){
    int u = blockIdx.x*blockDim.x + threadIdx.x; if (u>=K) return;
    int di = unused_idx[u];
    if (di < 0 || di >= D){
        atomicAdd(bad_count, 1);
        if (atomicCAS(first_bad_u, -1, u) == -2) {
            // no-op
        }
        // store first bad val if first_bad_u was -1 before
        int old = atomicCAS(first_bad_u, -1, u);
        if (old == -1 && first_bad_val){ *first_bad_val = di; }
    }
}

void validate_unused_idx(const int* unused_idx, int K, int D,
                         int* bad_count, int* first_bad_u, int* first_bad_val,
                         cudaStream_t stream){
    if (!unused_idx || K<=0){ if (bad_count) cudaMemsetAsync(bad_count, 0, sizeof(int), stream); if (first_bad_u) cudaMemsetAsync(first_bad_u, 0xFF, sizeof(int), stream); if (first_bad_val) cudaMemsetAsync(first_bad_val, 0xFF, sizeof(int), stream); return; }
    // init: bad_count=0, first_bad_u=-1, first_bad_val=-1
    if (bad_count) cudaMemsetAsync(bad_count, 0, sizeof(int), stream);
    if (first_bad_u) cudaMemsetAsync(first_bad_u, 0xFF, sizeof(int), stream);
    if (first_bad_val) cudaMemsetAsync(first_bad_val, 0xFF, sizeof(int), stream);
    int tb=256, gb=(K+tb-1)/tb; k_validate_unused_idx<<<gb,tb,0,stream>>>(unused_idx, K, D, bad_count, first_bad_u, first_bad_val);
}
__global__ void k_compact_tracks(int T, int max_age,
                    const float* kf_x, const float* kf_P,
                    const int* meta_id, const int* meta_cls, const float* meta_score, const int* meta_tsu, const int* meta_hit,
                    const float4* meta_obs,
                    const float2* obs_prev, const float2* obs_curr,
                    const int* trk_cls,
                    // dst
                    float* tmp_kf_x, float* tmp_kf_P,
                    int* tmp_meta_id, int* tmp_meta_cls, float* tmp_meta_score, int* tmp_meta_tsu, int* tmp_meta_hit,
                    float4* tmp_meta_obs,
                    float2* tmp_obs_prev, float2* tmp_obs_curr,
                    int* tmp_trk_cls,
                    int* newT){
    int j=blockIdx.x*blockDim.x+threadIdx.x; if (j>=T) return;
    if (meta_tsu[j] <= max_age){ int idx = atomicAdd(newT,1);
        // copy
        for (int k=0;k<7;++k) tmp_kf_x[idx*7+k]=kf_x[j*7+k];
        for (int k=0;k<49;++k) tmp_kf_P[idx*49+k]=kf_P[j*49+k];
        tmp_meta_id[idx]=meta_id[j]; tmp_meta_cls[idx]=meta_cls[j]; tmp_meta_score[idx]=meta_score[j]; tmp_meta_tsu[idx]=meta_tsu[j]; tmp_meta_hit[idx]=meta_hit[j];
        tmp_meta_obs[idx]=meta_obs[j]; tmp_obs_prev[idx]=obs_prev[j]; tmp_obs_curr[idx]=obs_curr[j]; tmp_trk_cls[idx]=trk_cls[j];
    }
}

void compact_tracks(int T, int max_age,
                    float* kf_x, float* kf_P,
                    int* meta_id, int* meta_cls, float* meta_score, int* meta_tsu, int* meta_hit,
                    float4* meta_obs,
                    float2* obs_prev, float2* obs_curr,
                    int* trk_cls,
                    float* tmp_kf_x, float* tmp_kf_P,
                    int* tmp_meta_id, int* tmp_meta_cls, float* tmp_meta_score, int* tmp_meta_tsu, int* tmp_meta_hit,
                    float4* tmp_meta_obs,
                    float2* tmp_obs_prev, float2* tmp_obs_curr,
                    int* tmp_trk_cls,
                    int* newT,
                    cudaStream_t stream){
    if (T<=0){ cudaMemsetAsync(newT, 0, sizeof(int), stream); return; }
    cudaMemsetAsync(newT, 0, sizeof(int), stream);
    int tb=256, gb=(T+tb-1)/tb;
    k_compact_tracks<<<gb,tb,0,stream>>>(T, max_age, kf_x, kf_P, meta_id, meta_cls, meta_score, meta_tsu, meta_hit, meta_obs, obs_prev, obs_curr, trk_cls,
                                         tmp_kf_x, tmp_kf_P, tmp_meta_id, tmp_meta_cls, tmp_meta_score, tmp_meta_tsu, tmp_meta_hit, tmp_meta_obs, tmp_obs_prev, tmp_obs_curr, tmp_trk_cls, newT);
}

void swap_compacted_back(
                    float*& kf_x, float*& kf_P,
                    int*& meta_id, int*& meta_cls, float*& meta_score, int*& meta_tsu, int*& meta_hit,
                    float4*& meta_obs,
                    float2*& obs_prev, float2*& obs_curr,
                    int*& trk_cls,
                    float*& tmp_kf_x, float*& tmp_kf_P,
                    int*& tmp_meta_id, int*& tmp_meta_cls, float*& tmp_meta_score, int*& tmp_meta_tsu, int*& tmp_meta_hit,
                    float4*& tmp_meta_obs,
                    float2*& tmp_obs_prev, float2*& tmp_obs_curr,
                    int*& tmp_trk_cls){
    // simple pointer swap
    auto swp=[&](auto*& a, auto*& b){ auto* t=a; a=b; b=t; };
    swp(kf_x, tmp_kf_x); swp(kf_P, tmp_kf_P);
    swp(meta_id, tmp_meta_id); swp(meta_cls, tmp_meta_cls); swp(meta_score, tmp_meta_score); swp(meta_tsu, tmp_meta_tsu); swp(meta_hit, tmp_meta_hit);
    swp(meta_obs, tmp_meta_obs);
    swp(obs_prev, tmp_obs_prev); swp(obs_curr, tmp_obs_curr);
    swp(trk_cls, tmp_trk_cls);
}

} // namespace ocs
