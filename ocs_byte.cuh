#pragma once
#include <cuda_runtime.h>

namespace ocs {

// Select best unmatched det for each unmatched track using IOU matrix on device.
// iou: N(det) x M(trk) matrix (row-major det-major)
// det_used: N, 0/1; trk_matched: M, 0/1
// sel_det: M, output selected det index or -1
void byte_select(const float* iou, int N, int M, float th,
                 const int* det_cls, const int* trk_cls,
                 int* det_used, int* trk_matched, int* sel_det,
                 cudaStream_t stream);

// Primary selection from Auction pairs on device with IOU/class/threshold filtering.
// pairs: device int[2*K], k_count: device int*
// det_cls[M], trk_cls[M]: device pointers, may be nullptr to disable class check
// tsu[M]: device time_since_update per track
// base_th: base IOU threshold, recover_relax in [0.5,1.0], ocr_enable toggles dynamic relax
void primary_select_from_pairs(const float* iou, int N, int M,
                              const int* pairs, const int* k_count,
                              const int* det_cls, const int* trk_cls,
                              const int* tsu,
                              float base_th, float recover_relax, int ocr_enable, float ocr_slope,
                              int* det_used, int* sel_det,
                              cudaStream_t stream);

// Build KF z/m from selected dets; prev_centers: M float2 (Y,X)
void build_z_from_sel(const float* det_xyxy, int N,
                      const int* sel_det, int M,
                      const float2* prev_centers,
                      float oru_alpha,
                      float* z_out, int* m_out,
                      cudaStream_t stream);

// Apply selection to meta arrays on device (update obs/score/cls for matched tracks)
void apply_sel_to_meta(const float* det_xyxy, const float* det_scores, const int* det_cls, int N,
                       const int* sel_det, int M,
                       float4* meta_obs, float* meta_score, int* meta_cls,
                       cudaStream_t stream);

// Mark trk_matched[tj]=1 if sel_det[tj]>=0 (device)
void sel_to_trk_matched(const int* sel_det, int M, int* trk_matched, cudaStream_t stream);

// Convert meta_obs (x1,y1,x2,y2) to centers (cy,cx)
void meta_obs_to_centers(const float4* meta_obs, int M, float2* out_curr, cudaStream_t stream);

// Build list of unused detections (det_used==0)
void build_unused_det_list(const int* det_used, int N, int* out_indices, int* out_count, cudaStream_t stream);

// Gather detection attributes by index list (device -> device compact)
void gather_dets_by_index(const float* det_xyxy, const float* det_scores, const int* det_cls,
                          const int* indices, int count,
                          float* out_xyxy4N, float* out_scores, int* out_cls,
                          cudaStream_t stream);

// Utility: zero/init arrays
void memset_int(int* ptr, int value, int count, cudaStream_t stream);
void memset_float(float* ptr, float value, int count, cudaStream_t stream);

} // namespace ocs
