#include "letterbox.hpp"
#include "letterbox_kernels_decl.cuh"
#include "letterbox_common.cuh"
#include "letterbox_launch_common.cuh"
#include "letterbox_kernels_templates.cuh"
#include "letterbox_lb_macros.cuh"

namespace trtyolo {

__global__ void halfToFloatKernel(const __half* src, float* dst, size_t count) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __half2float(src[i]);
}

void cudaHalfToFloat(const void* src, void* dst, size_t count, cudaStream_t stream) {
    constexpr unsigned int threads = 256;
    halfToFloatKernel<<<static_cast<unsigned int>((count + threads - 1) / threads), threads, 0, stream>>>(
        static_cast<const __half*>(src), static_cast<float*>(dst), count);
}

void cudaLetterboxHalf(const void* src, int src_cols, int src_rows, size_t src_pitch,
                       void* dst, int dst_cols, int dst_rows,
                       const LetterboxTransform& params, const ProcessConfig config, cudaStream_t stream) {
    dim3 block = pickBlock(config, dst_cols);
    dim3 grid(iDivUp((dst_cols + 3) / 4, block.x), iDivUp(dst_rows, block.y));
    if (config.use_launch_bounds) {
        int MINB = config.min_blocks_per_sm; MINB = (MINB < 1 ? 1 : (MINB > 3 ? 3 : MINB));
        LB_DISPATCH_TABLE_OR_GENERIC(LB_MINB_CALL_SINGLE, gpuLetterboxF16Vec4_lb,
            gpuLetterboxF16Vec4_generic<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config));
    } else {
        gpuLetterboxF16Vec4_generic<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config);
    }
}

} // namespace trtyolo
