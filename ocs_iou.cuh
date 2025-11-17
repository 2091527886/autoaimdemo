#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "ocs_state.cuh"

namespace ocs {

// Coordinate convention notes:
// - float2 direction/momentum uses (Y, X) order.
// - Centers are stored as (cy, cx) in float2.
// - Make sure any (cx,cy) pair converted into float2 keeps this (Y,X) mapping.

enum class Assoc : int { IOU=0, GIOU=1, CIOU=2, DIOU=3, CTDIST=4 };

// dets: N boxes, trks: M boxes
// out_cost: N*M row-major (det major), value = -(iou + vdc_weight*angle)
// If cate_mismatch: large negative penalty already applied outside
void iou_cost_matrix(const Box* dets, int N,
                     const Box* trks, int M,
                     const float* det_scores, // N
                     const float* trk_prev_obs_valid, // M (1 or 0)
                     const float2* vdc_dirs,  // M (inertia_y, inertia_x)
                     float vdc_weight,
                     Assoc ass_fn,
                     // optional class arrays (device) for masking; pass nullptr to disable
                     const int* det_cls,
                     const int* trk_cls,
                     int class_mask_enable,
                     float* out_iou,        // N*M iou-like score in [0,1]
                     float* out_angle,      // N*M [0,1]
                     float* out_cost,       // N*M = -(iou + vdc_weight*angle)*score (or large neg if class mismatched)
                     cudaStream_t stream);

}
