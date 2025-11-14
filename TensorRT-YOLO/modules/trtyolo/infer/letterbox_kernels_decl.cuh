#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "letterbox.hpp"
#include "letterbox_common.cuh"

namespace trtyolo {

// Declarations only; definitions are compiled in a single TU.
__global__ void gpuLetterboxF16Vec4_generic(const void* __restrict__ src, int src_cols, int src_rows, size_t src_pitch,
                                            void* __restrict__ dst, int dst_cols, int dst_rows,
                                            LetterboxTransform params, ProcessConfig config);

template<int BX, int BY, int MINB>
__global__ void gpuLetterboxF16Vec4_lb(const void* __restrict__ src, int src_cols, int src_rows, size_t src_pitch,
                                       void* __restrict__ dst, int dst_cols, int dst_rows,
                                       LetterboxTransform params, ProcessConfig config);

__global__ void gpuMultiLetterboxHalfVec4_generic(const void* __restrict__ src, int src_cols, int src_rows, size_t src_pitch,
                                                  void* __restrict__ dst, int dst_cols, int dst_rows,
                                                  LetterboxTransform params, ProcessConfig config,
                                                  int num_images);

template<int BX, int BY, int MINB>
__global__ void gpuMultiLetterboxHalfVec4_lb(const void* __restrict__ src, int src_cols, int src_rows, size_t src_pitch,
                                             void* __restrict__ dst, int dst_cols, int dst_rows,
                                             LetterboxTransform params, ProcessConfig config,
                                             int num_images);

} // namespace trtyolo
