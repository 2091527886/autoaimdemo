#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace ocs {

// Coordinate convention notes:
// - Box uses (x1,y1,x2,y2)
// - KF state stores (cx,cy,s,r)
// - Centers float2 are stored as (cy,cx) to align with (Y,X) usage in direction vectors

struct Box { float x1, y1, x2, y2; };

// Convert KF state (cx,cy,s,r,...) to xyxy boxes for T tracks
void kfx_to_boxes(const float* kfx, int T, Box* out_boxes, cudaStream_t stream);

// Age tsu by +1 for all tracks
void age_tsu_begin(int T, int* meta_tsu, cudaStream_t stream);

// After selection: for matched tracks (sel_det[j] >= 0), set tsu=0 and hit+=1
void update_hit_tsu_after_match(const int* sel_det, int T, int* meta_hit, int* meta_tsu, cudaStream_t stream);

// Append new tracks from unused detections list (device side)
// det_xyxy: Nx4, det_scores: N, det_cls: N
// unused_idx: K indices into det arrays
// trk_count: device scalar current track count, next_id: device scalar next id
void append_new_tracks(const float* det_xyxy, const float* det_scores, const int* det_cls,
                       const int* unused_idx, int K,
                       int cap_M,
                       int* trk_count, int* next_id,
                       // outputs / states
                       float* kf_x, float* kf_P,
                       int* meta_id, int* meta_cls, float* meta_score, int* meta_tsu, int* meta_hit,
                       float4* meta_obs,
                       float2* obs_prev, float2* obs_curr,
                       int* trk_cls,
                       cudaStream_t stream);

// Compact tracks by removing those with meta_tsu > max_age
// Writes compacted arrays into tmp buffers; newT (device scalar) is written with result
void compact_tracks(int T, int max_age,
                    // src
                    float* kf_x, float* kf_P,
                    int* meta_id, int* meta_cls, float* meta_score, int* meta_tsu, int* meta_hit,
                    float4* meta_obs,
                    float2* obs_prev, float2* obs_curr,
                    int* trk_cls,
                    // tmp dst
                    float* tmp_kf_x, float* tmp_kf_P,
                    int* tmp_meta_id, int* tmp_meta_cls, float* tmp_meta_score, int* tmp_meta_tsu, int* tmp_meta_hit,
                    float4* tmp_meta_obs,
                    float2* tmp_obs_prev, float2* tmp_obs_curr,
                    int* tmp_trk_cls,
                    int* newT,
                    cudaStream_t stream);

// Swap tmp back to main after compaction
void swap_compacted_back(
                    // main
                    float*& kf_x, float*& kf_P,
                    int*& meta_id, int*& meta_cls, float*& meta_score, int*& meta_tsu, int*& meta_hit,
                    float4*& meta_obs,
                    float2*& obs_prev, float2*& obs_curr,
                    int*& trk_cls,
                    // tmp
                    float*& tmp_kf_x, float*& tmp_kf_P,
                    int*& tmp_meta_id, int*& tmp_meta_cls, float*& tmp_meta_score, int*& tmp_meta_tsu, int*& tmp_meta_hit,
                    float4*& tmp_meta_obs,
                    float2*& tmp_obs_prev, float2*& tmp_obs_curr,
                    int*& tmp_trk_cls);

} // namespace ocs
