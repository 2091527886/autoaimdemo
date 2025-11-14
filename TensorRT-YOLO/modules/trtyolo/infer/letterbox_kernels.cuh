#pragma once
#include "letterbox_body.cuh"

namespace trtyolo {

// 通用版本（无 launch_bounds），F16 vectorized: 4 pixels/thread
__global__ void gpuLetterboxF16Vec4_generic(const void* __restrict__ src, int src_cols, int src_rows, size_t src_pitch,
                                    void* __restrict__ dst, int dst_cols, int dst_rows,
                                    LetterboxTransform params, ProcessConfig config) {
    letterboxF16Vec4_body(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
}

// 启用 launch_bounds 的模板内核
template<int BX, int BY, int MINB>
__launch_bounds__(BX*BY, MINB)
__global__ void gpuLetterboxF16Vec4_lb(const void* __restrict__ src, int src_cols, int src_rows, size_t src_pitch,
                                    void* __restrict__ dst, int dst_cols, int dst_rows,
                                    LetterboxTransform params, ProcessConfig config) {
    letterboxF16Vec4_body(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
}

// Vectorized multi-image generic
__global__ void gpuMultiLetterboxHalfVec4_generic(const void* __restrict__ src, int src_cols, int src_rows, size_t src_pitch,
                                          void* __restrict__ dst, int dst_cols, int dst_rows,
                                          LetterboxTransform params, ProcessConfig config,
                                          int num_images) {
    multiLetterboxHalfVec4_body(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
}

// 启用 launch_bounds 的模板多图内核
template<int BX, int BY, int MINB>
__launch_bounds__(BX*BY, MINB)
__global__ void gpuMultiLetterboxHalfVec4_lb(const void* __restrict__ src, int src_cols, int src_rows, size_t src_pitch,
                                          void* __restrict__ dst, int dst_cols, int dst_rows,
                                          LetterboxTransform params, ProcessConfig config,
                                          int num_images) {
    multiLetterboxHalfVec4_body(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
}

} // namespace trtyolo
