#include "letterbox.hpp"
#include "letterbox_kernels_decl.cuh"
#include "letterbox_common.cuh"
#include "letterbox_launch_common.cuh"
#include "letterbox_kernels_templates.cuh"

namespace trtyolo {

void cudaMultiLetterboxHalf(const void* src, int src_cols, int src_rows, size_t src_pitch,
                            void* dst, int dst_cols, int dst_rows,
                            const LetterboxTransform& params, const ProcessConfig config,
                            int num_images, cudaStream_t stream) {
    dim3 block = pickBlock(config, dst_cols);
    dim3 grid(iDivUp((dst_cols + 3) / 4, block.x), iDivUp(dst_rows, block.y), num_images);
    if (config.use_launch_bounds) {
        int MINB = config.min_blocks_per_sm; MINB = (MINB < 1 ? 1 : (MINB > 3 ? 3 : MINB));
        if (block.x == 32 && block.y == 8) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<32,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<32,8,2><<<grid, block, 0, stream>>> (src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<32,8,3><<<grid, block, 0, stream>>> (src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 16 && block.y == 16) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<16,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<16,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<16,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 64 && block.y == 4) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<64,4,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<64,4,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<64,4,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 32 && block.y == 16) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<32,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<32,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<32,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 64 && block.y == 8) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<64,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<64,8,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<64,8,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 48 && block.y == 8) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<48,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<48,8,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<48,8,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 24 && block.y == 16) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<24,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<24,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<24,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 40 && block.y == 8) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<40,8,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<40,8,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<40,8,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else if (block.x == 20 && block.y == 16) {
            if (MINB<=1)      gpuMultiLetterboxHalfVec4_lb<20,16,1><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else if (MINB==2) gpuMultiLetterboxHalfVec4_lb<20,16,2><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
            else              gpuMultiLetterboxHalfVec4_lb<20,16,3><<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        } else {
            gpuMultiLetterboxHalfVec4_generic<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
        }
    } else {
        gpuMultiLetterboxHalfVec4_generic<<<grid, block, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, params, config, num_images);
    }
}

} // namespace trtyolo
